"""Teacher signals for distillation — the per-frame quality dimensions to mimic.

Four come from the existing classical OpenCV metrics (essentially free); the fifth,
learned IQA, is the genuinely expensive deep teacher (pyiqa TOPIQ, a ResNet-50
network) that the compact student is meant to replace. All targets are on the
project's 0–100, higher-is-better convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from video_benchmark.metrics.anomalies import AnomalyDetector
from video_benchmark.metrics.blur import BlurClassifier
from video_benchmark.metrics.brightness import BrightnessMetric
from video_benchmark.metrics.sharpness import SharpnessMetric

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Output order of the student's heads. Keep stable — checkpoints depend on it.
TARGETS: list[str] = ["brightness", "sharpness", "blur", "anomaly", "iqa"]

# The classical OpenCV signals (free) vs. the deep signal we actually distil.
_CLASSICAL = ["brightness", "sharpness", "blur", "anomaly"]


def _norm_sharpness(lap_var: float) -> float:
    """Match scoring._normalize_sharpness (per frame)."""
    return float(min(100.0, lap_var / 5.0))


class TeacherLabeler:
    """Produce the per-frame target vector used to supervise the student."""

    def __init__(self, device: str = "cpu", iqa_model: str = "topiq_nr") -> None:
        self._brightness = BrightnessMetric()
        self._sharpness = SharpnessMetric()
        self._blur = BlurClassifier()
        self._anomaly = AnomalyDetector()
        self._device = device
        self._iqa_model = iqa_model
        self._iqa: object | None = None  # lazy — only load pyiqa when needed

    # --- classical (cheap) -------------------------------------------------

    def classical(self, frame: np.ndarray) -> dict[str, float]:
        """The four OpenCV signals for one BGR frame."""
        return {
            "brightness": self._brightness.normalize(self._brightness.compute(frame)),
            "sharpness": _norm_sharpness(self._sharpness.compute(frame)),
            "blur": max(0.0, 100.0 - self._blur.classify(frame).severity),
            "anomaly": self._anomaly.compute_anomaly_score(frame),
        }

    # --- learned IQA (expensive deep teacher) ------------------------------

    def _ensure_iqa(self) -> object:
        if self._iqa is None:
            import pyiqa

            logger.info("Loading pyiqa teacher %s on %s", self._iqa_model, self._device)
            self._iqa = pyiqa.create_metric(self._iqa_model, device=self._device)
        return self._iqa

    def iqa_batch(self, batch: torch.Tensor) -> list[float]:
        """Run the IQA teacher on a normalized [0,1] RGB tensor batch (N,3,H,W)."""
        import torch

        metric = self._ensure_iqa()
        with torch.no_grad():
            scores = metric(batch.to(self._device))  # type: ignore[operator]
        arr = scores.detach().cpu().reshape(-1).numpy() * 100.0
        return [float(x) for x in arr]

    def label_frames(
        self,
        frames: list[np.ndarray],
        iqa_batch: torch.Tensor,
    ) -> np.ndarray:
        """Return an (N, len(TARGETS)) target matrix for a batch of frames.

        ``iqa_batch`` is the same frames preprocessed to pyiqa's expected
        [0,1] RGB tensor (built by the dataset to avoid double work).
        """
        iqa_scores = self.iqa_batch(iqa_batch)
        rows = []
        for frame, iqa in zip(frames, iqa_scores, strict=True):
            c = self.classical(frame)
            rows.append([c["brightness"], c["sharpness"], c["blur"], c["anomaly"], iqa])
        return np.asarray(rows, dtype=np.float32)
