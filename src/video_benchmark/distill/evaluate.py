"""Evaluation + head-to-head benchmark: fidelity, latency, throughput, size."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from video_benchmark.distill.data import (
    FeatureCache,
    _pyiqa_tensor,
    build_backbone_transform,
    sample_frames,
)
from video_benchmark.distill.model import CompactQualityNet
from video_benchmark.distill.teacher import TARGETS, TeacherLabeler


def _corr(a: np.ndarray, b: np.ndarray, kind: str = "plcc") -> float:
    """Pearson/Spearman correlation, NaN when a column is constant."""
    if a.std() <= 1e-6 or b.std() <= 1e-6:
        return float("nan")
    fn = pearsonr if kind == "plcc" else spearmanr
    return float(fn(a, b)[0])


@dataclass
class Fidelity:
    per_target_plcc: dict[str, float]
    per_target_srcc: dict[str, float]
    per_target_mae: dict[str, float]  # 0..100 units
    composite_plcc: float = float("nan")  # agreement on the overall quality verdict
    composite_srcc: float = float("nan")

    @property
    def mean_plcc(self) -> float:
        return float(np.nanmean(list(self.per_target_plcc.values())))

    @property
    def mean_srcc(self) -> float:
        return float(np.nanmean(list(self.per_target_srcc.values())))


@dataclass
class BenchResult:
    student_ms_per_frame: float
    teacher_ms_per_frame: float
    student_fps_batched: float
    teacher_fps: float
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def speedup_latency(self) -> float:
        return self.teacher_ms_per_frame / max(1e-9, self.student_ms_per_frame)

    @property
    def speedup_throughput(self) -> float:
        return self.student_fps_batched / max(1e-9, self.teacher_fps)


def evaluate_fidelity(
    model: CompactQualityNet,
    cache: FeatureCache,
    val_idx: np.ndarray,
    device: str,
) -> Fidelity:
    """Correlation of student vs. teacher on held-out frames, per target."""
    feats = torch.from_numpy(cache.features[val_idx]).float().to(device)
    y_true = cache.targets[val_idx]
    model.trunk.eval()
    with torch.no_grad():
        y_pred = model.forward_from_features(feats).cpu().numpy()

    plcc, srcc, mae = {}, {}, {}
    for i, name in enumerate(TARGETS):
        t, p = y_true[:, i], y_pred[:, i]
        plcc[name] = _corr(t, p, "plcc")
        srcc[name] = _corr(t, p, "srcc")
        mae[name] = float(np.mean(np.abs(t - p)))

    # Composite = mean over targets that genuinely vary; the bottom-line "do the
    # student and teacher agree on overall frame quality" number.
    varying = [i for i in range(len(TARGETS)) if y_true[:, i].std() > 1e-6]
    ct, cp = y_true[:, varying].mean(axis=1), y_pred[:, varying].mean(axis=1)
    return Fidelity(
        plcc, srcc, mae, composite_plcc=_corr(ct, cp, "plcc"), composite_srcc=_corr(ct, cp, "srcc")
    )


def benchmark_speed(
    model: CompactQualityNet,
    teacher: TeacherLabeler,
    sample_video: str,
    device: str,
    batch_size: int = 32,
    iters: int = 40,
) -> BenchResult:
    """Compare the student (one forward) against the teacher stack (pyiqa + CV)."""
    transform = build_backbone_transform(model.backbone)

    frames = sample_frames(sample_video, fps_sample=4.0, max_frames=batch_size)
    if not frames:
        frames = [np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)]
    frames = (frames * (batch_size // len(frames) + 1))[:batch_size]

    pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    x = torch.stack([transform(p) for p in pil]).to(device)
    model.eval().to(device)

    def sync() -> None:
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    # --- student: single-frame latency ---
    x1 = x[:1]
    with torch.no_grad():
        for _ in range(5):
            model(x1)
        sync()
        t = time.perf_counter()
        for _ in range(iters):
            model(x1)
        sync()
        student_ms = (time.perf_counter() - t) / iters * 1000.0

        # --- student: batched throughput ---
        for _ in range(3):
            model(x)
        sync()
        t = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync()
        student_fps = batch_size * iters / (time.perf_counter() - t)

    # --- teacher: per-frame = pyiqa (deep) + classical CV ---
    iqa_in = _pyiqa_tensor(frames[:1])
    teacher.iqa_batch(iqa_in)  # warmup/load
    t = time.perf_counter()
    reps = max(5, iters // 4)
    for _ in range(reps):
        teacher.iqa_batch(iqa_in)
        teacher.classical(frames[0])
    teacher_ms = (time.perf_counter() - t) / reps * 1000.0
    teacher_fps = 1000.0 / teacher_ms

    return BenchResult(
        student_ms_per_frame=student_ms,
        teacher_ms_per_frame=teacher_ms,
        student_fps_batched=student_fps,
        teacher_fps=teacher_fps,
    )


def model_size_mb(model: CompactQualityNet) -> dict[str, float]:
    """Parameter footprint at fp32 / fp16 / int8."""
    n = sum(p.numel() for p in model.parameters())
    return {
        "params_millions": n / 1e6,
        "fp32_mb": n * 4 / 1e6,
        "fp16_mb": n * 2 / 1e6,
        "int8_mb": n * 1 / 1e6,
    }
