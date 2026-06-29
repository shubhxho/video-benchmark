"""Run the full distillation: label -> train -> evaluate -> benchmark -> export.

Usage:
    uv run python -m video_benchmark.distill --videos videos --epochs 300
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import torch
from rich.console import Console
from rich.text import Text

from video_benchmark.distill import ui
from video_benchmark.distill.data import build_feature_cache, split_by_clip
from video_benchmark.distill.evaluate import (
    BenchResult,
    Fidelity,
    benchmark_speed,
    evaluate_fidelity,
    model_size_mb,
)
from video_benchmark.distill.model import DEFAULT_BACKBONE, CompactQualityNet
from video_benchmark.distill.teacher import DEEP, TARGETS, TeacherLabeler
from video_benchmark.distill.train import train_heads

console = Console()
# Progress/banner go here; in --emit-json mode it is redirected to stderr so
# stdout carries only the machine-readable JSON the Go front-end consumes.
progress_console = console

_NOISY_LOGGERS = (
    "huggingface_hub",
    "timm",
    "open_clip",
    "pyiqa",
    "transformers",
    "PIL",
    "urllib3",
    "filelock",
    "fsspec",
    "video_benchmark",
)


def _silence_libraries() -> None:
    """Mute third-party warnings/logging so only the charm UI shows."""
    for key, val in {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(key, val)
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)  # drop INFO/WARNING from every library
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Swallow stray stdout/stderr prints from libraries during heavy phases.

    The module-level Rich ``console`` keeps the real stdout, so our own output is
    unaffected; only library ``print`` calls land in the void.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _find_videos(root: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def _step(msg: str) -> None:
    progress_console.print(Text("  ▸ ", style=ui.PINK) + Text(msg, style=ui.MUTED))


def _results_dict(
    args: argparse.Namespace,
    device: str,
    n_clips: int,
    cache_len: int,
    n_train: int,
    n_val: int,
    fid: Fidelity,
    size: dict[str, float],
    bench: BenchResult,
    hist: dict[str, float],
    out_path: Path,
    on_disk: float,
) -> dict[str, object]:
    """Machine-readable results for the Go charm front-end (and JSON dumps)."""

    def nn(x: float) -> float | None:
        """NaN -> None so the output is valid JSON the Go parser accepts."""
        return None if math.isnan(x) else x

    return {
        "backbone": args.backbone.split(":")[-1],
        "device": device,
        "clips": n_clips,
        "fps": args.fps,
        "epochs": args.epochs,
        "frames": cache_len,
        "train": n_train,
        "val": n_val,
        "targets": [
            {
                "name": name,
                "kind": "deep" if name in DEEP else "cv",
                "std": fid.per_target_std[name],
                "plcc": nn(fid.per_target_plcc[name]),
                "srcc": nn(fid.per_target_srcc[name]),
                "mae": fid.per_target_mae[name],
            }
            for name in TARGETS
        ],
        "composite_plcc": nn(fid.composite_plcc),
        "composite_srcc": nn(fid.composite_srcc),
        "deep_plcc": nn(fid.deep_plcc),
        "size": {
            "params_millions": size["params_millions"],
            "fp16_mb": size["fp16_mb"],
            "int8_mb": size["int8_mb"],
        },
        "speed": {
            "student_fps": bench.student_fps_batched,
            "teacher_fps": bench.teacher_fps,
            "student_ms": bench.student_ms_per_frame,
            "teacher_ms": bench.teacher_ms_per_frame,
            "speedup_throughput": bench.speedup_throughput,
            "speedup_latency": bench.speedup_latency,
        },
        "export": {
            "path": str(out_path),
            "mb": on_disk,
            "under_30mb": on_disk < 30,
            "best_val_loss": hist["best_val_loss"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Distil the metric stack into a compact model.")
    ap.add_argument("--videos", type=Path, default=Path("videos"))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fps", type=float, default=2.0, help="frames/sec sampled per clip")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("models/compact_quality.pt"))
    ap.add_argument(
        "--emit-json",
        action="store_true",
        help="print results as JSON to stdout (progress to stderr) for the Go front-end",
    )
    args = ap.parse_args()

    global progress_console
    if args.emit_json:
        progress_console = Console(stderr=True)

    _silence_libraries()
    device = _device(args.device)
    videos = _find_videos(args.videos)
    if not videos:
        progress_console.print(
            Text("✗ no videos found under ", style=ui.RED) + Text(str(args.videos))
        )
        raise SystemExit(1)

    progress_console.print()
    progress_console.print(
        ui.banner(
            "COMPACT QUALITY MODEL · distillation",
            f"{args.backbone.split(':')[-1]}   ·   {device}   ·   "
            f"{len(videos)} clips   ·   {args.fps} fps   ·   {args.epochs} epochs\n"
            f"targets   {' '.join(TARGETS)}",
        )
    )

    _step("loading MobileCLIP-S0 backbone + teachers")
    with _quiet():
        model = CompactQualityNet(
            backbone_name=args.backbone, pretrained=True, freeze_backbone=True
        ).to(device)
        teacher = TeacherLabeler(device=device)

    _step("sampling frames + running teachers (classical CV · pyiqa IQA · MobileCLIP scene)")
    with _quiet():
        cache = build_feature_cache(videos, model, teacher, device, fps_sample=args.fps)
    train_idx, val_idx = split_by_clip(cache)
    _step(
        f"frames={len(cache)}  train={int(train_idx.sum())}  val={int(val_idx.sum())}  "
        "(leakage-free per-clip split)"
    )

    _step("training student heads + benchmarking")
    with _quiet():
        hist = train_heads(model, cache, train_idx, val_idx, device, epochs=args.epochs, lr=args.lr)
        fid = evaluate_fidelity(model, cache, val_idx, device)
        size = model_size_mb(model)
        bench = benchmark_speed(model, teacher, str(videos[0]), device)

    # --- export (fp16) ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone_name": args.backbone,
        "targets": TARGETS,
        "state_dict": {k: v.half() for k, v in model.state_dict().items()},
    }
    torch.save(payload, args.out)
    on_disk = args.out.stat().st_size / 1e6

    if args.emit_json:
        results = _results_dict(
            args, device, len(videos), len(cache), int(train_idx.sum()),
            int(val_idx.sum()), fid, size, bench, hist, args.out, on_disk,
        )
        sys.stdout.write(json.dumps(results) + "\n")
        sys.stdout.flush()
    else:
        _render(fid, size, bench, hist, args.out, on_disk)


def _render(
    fid: Fidelity,
    size: dict[str, float],
    bench: BenchResult,
    hist: dict[str, float],
    out_path: Path,
    on_disk: float,
) -> None:
    # --- headline cards -----------------------------------------------------
    console.print()
    cards = [
        ui.stat_card(
            "COMPOSITE",
            f"{fid.composite_plcc:+.2f}",
            "verdict agreement",
            ui._corr_color(fid.composite_plcc),
        ),
        ui.stat_card(
            "DEEP SIGNALS",
            f"{fid.deep_plcc:+.2f}",
            "iqa + scene fidelity",
            ui._corr_color(fid.deep_plcc),
        ),
        ui.stat_card(
            "THROUGHPUT",
            f"{bench.speedup_throughput:.1f}×",
            f"{bench.student_fps_batched:.0f} vs {bench.teacher_fps:.0f} fps",
            ui.GREEN,
        ),
        ui.stat_card(
            "SIZE",
            f"{size['fp16_mb']:.0f} MB",
            f"fp16 · int8 {size['int8_mb']:.0f}MB",
            ui.GREEN if size["fp16_mb"] < 30 else ui.RED,
        ),
    ]
    console.print(ui.headline_cards(cards))

    # --- fidelity table -----------------------------------------------------
    console.print()
    console.print(ui.rule("FIDELITY · student reproduces the teacher"))
    tbl = ui.clean_table()
    tbl.add_column("signal", no_wrap=True)
    tbl.add_column("kind", justify="center", width=6)
    tbl.add_column("spread", justify="right", width=8)
    tbl.add_column("agreement (PLCC)", width=26)
    tbl.add_column("MAE", justify="right", width=7)
    for name in TARGETS:
        is_deep = name in DEEP
        sig = Text(("◆ " if is_deep else "  ") + name, style=ui.VIOLET if is_deep else ui.MUTED)
        kind = ui.badge("deep", ui.VIOLET) if is_deep else Text("cv", style=ui.FAINT)
        std = fid.per_target_std[name]
        spread = Text(f"{std:.0f}" + ("·flat" if std < 5 else ""), style=ui.FAINT)
        mae = fid.per_target_mae[name]
        mae_color = ui.GREEN if mae < 10 else ui.AMBER if mae < 20 else ui.RED
        tbl.add_row(
            sig,
            kind,
            spread,
            ui.gauge(fid.per_target_plcc[name]),
            Text(f"{mae:.1f}", style=mae_color),
        )
    console.print(ui.panelize("per-signal  (deep = distilled · cv = exact, kept in OpenCV)", tbl))

    # --- speed --------------------------------------------------------------
    console.print()
    console.print(ui.rule("SPEED · the time-complexity win"))
    maxfps = max(bench.student_fps_batched, bench.teacher_fps, 1.0)

    def fps_row(label: str, fps: float, color: str) -> Text:
        n = round(fps / maxfps * 22)
        line = Text(f"{label:<9}", style=ui.MUTED)
        line.append(ui.FILLED * n, style=color)
        line.append(ui.EMPTY * (22 - n), style=ui.FAINT)
        line.append(f"  {fps:.0f} fps", style=color)
        return line

    speed = Text()
    speed.append_text(fps_row("student", bench.student_fps_batched, ui.GREEN))
    speed.append("\n")
    speed.append_text(fps_row("teacher", bench.teacher_fps, ui.AMBER))
    speed.append("\n\n")
    speed.append("one forward pass replaces pyiqa + MobileCLIP scene + CV  →  ", style=ui.MUTED)
    speed.append(f"▲ {bench.speedup_throughput:.1f}× faster", style=f"bold {ui.GREEN}")
    speed.append(
        f"   ({bench.student_ms_per_frame:.1f} vs {bench.teacher_ms_per_frame:.1f} ms/frame)",
        style=ui.FAINT,
    )
    console.print(ui.panelize("throughput  ·  student vs teacher stack", speed, accent=ui.GREEN))

    # --- export footer ------------------------------------------------------
    console.print()
    ok = on_disk < 30
    foot = Text()
    foot.append("✓ saved  " if ok else "✗ saved  ", style=ui.GREEN if ok else ui.RED)
    foot.append(str(out_path), style="bold")
    foot.append(f"   {on_disk:.1f} MB fp16", style=ui.MUTED)
    foot.append("\n")
    foot.append("under-30MB target  ", style=ui.MUTED)
    foot.append("PASS" if ok else "FAIL", style=f"bold {ui.GREEN if ok else ui.RED}")
    foot.append(f"     best val loss {hist['best_val_loss']:.4f}", style=ui.FAINT)
    console.print(ui.panelize("export", foot, accent=ui.PINK))
    console.print()


if __name__ == "__main__":
    main()
