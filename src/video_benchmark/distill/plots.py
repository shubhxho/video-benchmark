"""Generate charm-styled chart PNGs for the report / Hugging Face model card."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from video_benchmark.distill.evaluate import BenchResult, Fidelity  # noqa: E402
from video_benchmark.distill.teacher import DEEP, TARGETS  # noqa: E402
from video_benchmark.distill.train import TrainHistory  # noqa: E402

# charm/lipgloss palette
BG = "#0E0E16"
PANEL = "#1A1A24"
PINK = "#FF6AC1"
VIOLET = "#B794F6"
GREEN = "#74E39B"
AMBER = "#F6C177"
TEXT = "#C8C8D4"
MUTED = "#7A7C90"


def _style(ax: Axes, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=PINK, fontweight="bold", fontsize=12, loc="left", pad=12)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.grid(True, color="#2A2A38", linewidth=0.6)


def _fig() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    fig.patch.set_facecolor(BG)
    return fig, ax


def loss_curve(hist: TrainHistory, out: Path) -> Path:
    fig, ax = _fig()
    _style(ax, "Training — distillation loss")
    epochs = range(1, len(hist.train_losses) + 1)
    ax.plot(epochs, hist.train_losses, color=VIOLET, lw=2, label="train")
    ax.plot(epochs, hist.val_losses, color=GREEN, lw=2, label="val")
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel("SmoothL1 loss", color=MUTED)
    leg = ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    leg.get_frame().set_alpha(0.8)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def fidelity_bars(fid: Fidelity, out: Path) -> Path:
    fig, ax = _fig()
    _style(ax, "Fidelity — student vs teacher (PLCC)")
    names, vals, colors = [], [], []
    for name in TARGETS:
        v = fid.per_target_plcc[name]
        if v != v:  # NaN → low-variance, skip
            continue
        names.append(name)
        vals.append(v)
        colors.append(VIOLET if name in DEEP else MUTED)
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y, names)
    ax.set_xlim(-1, 1)
    ax.axvline(0, color=MUTED, lw=0.8)
    for yi, v in zip(y, vals, strict=True):
        ax.text(v, float(yi), f" {v:+.2f}", va="center", color=TEXT, fontsize=9)
    ax.set_xlabel("Pearson correlation (deep = violet)", color=MUTED)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def scatter_iqa(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> Path:
    idx = TARGETS.index("iqa")
    t, p = y_true[:, idx], y_pred[:, idx]
    fig, ax = _fig()
    _style(ax, "Learned IQA — student vs teacher")
    ax.scatter(t, p, color=PINK, alpha=0.7, edgecolor="none", s=28)
    lim = (float(min(t.min(), p.min())) - 2, float(max(t.max(), p.max())) + 2)
    ax.plot(lim, lim, color=MUTED, ls="--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("teacher (pyiqa TOPIQ)", color=MUTED)
    ax.set_ylabel("student", color=MUTED)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def speed_bars(bench: BenchResult, out: Path) -> Path:
    fig, ax = _fig()
    _style(ax, "Throughput — one forward pass vs teacher stack")
    labels = ["student", "teacher"]
    vals = [bench.student_fps_batched, bench.teacher_fps]
    ax.bar(labels, vals, color=[GREEN, AMBER], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f} fps", ha="center", va="bottom", color=TEXT, fontsize=10)
    ax.set_ylabel("frames / sec", color=MUTED)
    ax.set_title(
        f"Throughput — {bench.speedup_throughput:.1f}× faster",
        color=PINK, fontweight="bold", fontsize=12, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def generate_all(
    out_dir: Path,
    hist: TrainHistory,
    fid: Fidelity,
    bench: BenchResult,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        loss_curve(hist, out_dir / "training_loss.png"),
        fidelity_bars(fid, out_dir / "fidelity.png"),
        scatter_iqa(y_true, y_pred, out_dir / "scatter_iqa.png"),
        speed_bars(bench, out_dir / "throughput.png"),
    ]
