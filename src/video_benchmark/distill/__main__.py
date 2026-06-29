"""Run the full distillation: label -> train -> evaluate -> benchmark -> export.

Usage:
    uv run python -m video_benchmark.distill --videos videos --epochs 300
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from video_benchmark.distill.data import build_feature_cache, split_by_clip
from video_benchmark.distill.evaluate import (
    benchmark_speed,
    evaluate_fidelity,
    model_size_mb,
)
from video_benchmark.distill.model import DEFAULT_BACKBONE, CompactQualityNet
from video_benchmark.distill.teacher import TARGETS, TeacherLabeler
from video_benchmark.distill.train import train_heads

console = Console()


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Distil the metric stack into a compact model.")
    ap.add_argument("--videos", type=Path, default=Path("videos"))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fps", type=float, default=2.0, help="frames/sec sampled per clip")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("models/compact_quality.pt"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = _device(args.device)
    videos = _find_videos(args.videos)
    if not videos:
        console.print(f"[red]No videos found under {args.videos}[/red]")
        raise SystemExit(1)

    console.print(
        Panel(
            f"backbone=[b]{args.backbone}[/b]\ndevice=[b]{device}[/b]  "
            f"clips=[b]{len(videos)}[/b]  fps={args.fps}  epochs={args.epochs}\n"
            f"targets: {', '.join(TARGETS)}",
            title="[bold]Distillation run[/bold]",
            border_style="blue",
        )
    )

    # Frozen backbone + trained heads (linear-probe distillation). This is the
    # robust choice for small corpora; for a large operator-video dataset, the
    # next step is to unfreeze the backbone for higher fidelity.
    model = CompactQualityNet(
        backbone_name=args.backbone, pretrained=True, freeze_backbone=True
    ).to(device)
    teacher = TeacherLabeler(device=device)

    console.print("Building feature cache (sampling frames, running teacher)…")
    cache = build_feature_cache(videos, model, teacher, device, fps_sample=args.fps)
    train_idx, val_idx = split_by_clip(cache)
    console.print(
        f"frames: [b]{len(cache)}[/b]  train=[b]{int(train_idx.sum())}[/b]  "
        f"val=[b]{int(val_idx.sum())}[/b]"
    )

    console.print("Training student heads…")
    hist = train_heads(model, cache, train_idx, val_idx, device, epochs=args.epochs, lr=args.lr)

    fid = evaluate_fidelity(model, cache, val_idx, device)
    size = model_size_mb(model)
    bench = benchmark_speed(model, teacher, str(videos[0]), device)

    # --- report ---
    ftab = Table(title="Distillation fidelity (student vs teacher, held-out frames)")
    ftab.add_column("Target")
    ftab.add_column("PLCC", justify="right")
    ftab.add_column("SRCC", justify="right")
    ftab.add_column("MAE (0-100)", justify="right")
    for name in TARGETS:
        ftab.add_row(
            name,
            f"{fid.per_target_plcc[name]:.3f}",
            f"{fid.per_target_srcc[name]:.3f}",
            f"{fid.per_target_mae[name]:.2f}",
        )
    ftab.add_row("[b]mean[/b]", f"[b]{fid.mean_plcc:.3f}[/b]", f"[b]{fid.mean_srcc:.3f}[/b]", "")
    ftab.add_row(
        "[b]composite[/b]",
        f"[b]{fid.composite_plcc:.3f}[/b]",
        f"[b]{fid.composite_srcc:.3f}[/b]",
        "",
    )
    console.print(ftab)

    stab = Table(title="Size & speed (the time-complexity win)")
    stab.add_column("Metric")
    stab.add_column("Value", justify="right")
    stab.add_row("params", f"{size['params_millions']:.2f} M")
    stab.add_row("size fp16 / int8", f"{size['fp16_mb']:.1f} / {size['int8_mb']:.1f} MB")
    stab.add_row("student latency", f"{bench.student_ms_per_frame:.2f} ms/frame")
    stab.add_row("teacher latency", f"{bench.teacher_ms_per_frame:.2f} ms/frame")
    stab.add_row("[b]latency speedup[/b]", f"[b]{bench.speedup_latency:.1f}×[/b]")
    stab.add_row("student throughput (batched)", f"{bench.student_fps_batched:.0f} fps")
    stab.add_row("teacher throughput", f"{bench.teacher_fps:.0f} fps")
    stab.add_row("[b]throughput speedup[/b]", f"[b]{bench.speedup_throughput:.1f}×[/b]")
    console.print(stab)

    # --- export (fp16 state dict) ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone_name": args.backbone,
        "targets": TARGETS,
        "state_dict": {k: v.half() for k, v in model.state_dict().items()},
    }
    torch.save(payload, args.out)
    on_disk = args.out.stat().st_size / 1e6
    console.print(
        Panel(
            f"saved [b]{args.out}[/b]  ([b]{on_disk:.1f} MB[/b] on disk, fp16)\n"
            f"best val loss={hist['best_val_loss']:.4f}  epochs={hist['epochs_run']}\n"
            f"under-30MB target: {'[green]PASS[/green]' if on_disk < 30 else '[red]FAIL[/red]'}",
            title="[bold]Export[/bold]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
