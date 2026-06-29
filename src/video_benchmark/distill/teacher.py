"""Teacher signals for distillation — the per-frame quality dimensions to mimic.

Four come from the existing classical OpenCV metrics (essentially free); the fifth,
learned IQA, is the genuinely expensive deep teacher (pyiqa TOPIQ, a ResNet-50
network) that the compact student is meant to replace. All targets are on the
project's 0–100, higher-is-better convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from video_benchmark.metrics.anomalies import AnomalyDetector
from video_benchmark.metrics.blur import BlurClassifier
from video_benchmark.metrics.brightness import BrightnessMetric
from video_benchmark.metrics.sharpness import SharpnessMetric

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Output order of the student's heads. Keep stable — checkpoints depend on it.
# The two deep signals (iqa, scene) are the ones a learned model truly earns;
# the others are exact OpenCV stats included for a unified multi-task readout.
TARGETS: list[str] = ["brightness", "sharpness", "blur", "anomaly", "iqa", "scene"]

# The classical OpenCV signals (free) vs. the deep signals we actually distil.
_CLASSICAL = ["brightness", "sharpness", "blur", "anomaly"]
# Signals that genuinely require a model (the student's real job). The rest are
# exact CV, shown for completeness but better computed directly at inference.
DEEP = ["iqa", "scene"]

# Newest, smallest open_clip MobileCLIP for the zero-shot scene teacher.
SCENE_TEACHER_MODEL = "MobileCLIP2-S0"
SCENE_TEACHER_PRETRAINED = "dfndr2b"

# Robotics-operator scene prompts: is the workspace/hands actually usable in view?
_SCENE_GOOD = [
    "a clear first-person view of hands working on a task",
    "an operator's point-of-view of tools and a workspace",
    "a sharp, well-lit view of a manipulation task",
]
_SCENE_BAD = [
    "a blurry or obstructed camera view",
    "a view of the ceiling, floor, or an empty wall",
    "a dark, unusable video frame",
]


class SceneTeacher:
    """Zero-shot 'is this a usable operator view' score (0..100) via MobileCLIP."""

    def __init__(self, device: str = "cpu") -> None:
        import open_clip
        import torch

        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            SCENE_TEACHER_MODEL, pretrained=SCENE_TEACHER_PRETRAINED
        )
        self._model.eval().to(device)
        tokenizer = open_clip.get_tokenizer(SCENE_TEACHER_MODEL)
        with torch.no_grad():
            tokens = tokenizer(_SCENE_GOOD + _SCENE_BAD).to(device)
            tf = self._model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        self._text_feats = tf
        self._n_good = len(_SCENE_GOOD)
        self._logit_scale = float(self._model.logit_scale.exp().item())

    def score(self, frames: list[np.ndarray]) -> list[float]:
        import torch
        from PIL import Image

        imgs = torch.stack(
            [
                self._preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in frames
            ]
        ).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = self._logit_scale * feats @ self._text_feats.T
            good = logits[:, : self._n_good].mean(dim=1)
            bad = logits[:, self._n_good :].mean(dim=1)
            p_good = torch.softmax(torch.stack([good, bad], dim=1), dim=1)[:, 0]
        return [float(x) * 100.0 for x in p_good.detach().cpu().numpy()]


def _norm_sharpness(lap_var: float) -> float:
    """Match scoring._normalize_sharpness (per frame)."""
    return float(min(100.0, lap_var / 5.0))


class TeacherLabeler:
    """Produce the per-frame target vector used to supervise the student."""

    def __init__(
        self, device: str = "cpu", iqa_model: str = "topiq_nr", use_scene: bool = True
    ) -> None:
        self._brightness = BrightnessMetric()
        self._sharpness = SharpnessMetric()
        self._blur = BlurClassifier()
        self._anomaly = AnomalyDetector()
        self._device = device
        self._iqa_model = iqa_model
        self._iqa: object | None = None  # lazy — only load pyiqa when needed
        self._use_scene = use_scene
        self._scene: SceneTeacher | None = None  # lazy — load MobileCLIP when needed

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

    def _ensure_scene(self) -> SceneTeacher:
        if self._scene is None:
            logger.info("Loading scene teacher %s on %s", SCENE_TEACHER_MODEL, self._device)
            self._scene = SceneTeacher(device=self._device)
        return self._scene

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
        if self._use_scene:
            scene_scores = self._ensure_scene().score(frames)
        else:
            scene_scores = [75.0] * len(frames)  # neutral fallback
        rows = []
        for frame, iqa, scene in zip(frames, iqa_scores, scene_scores, strict=True):
            c = self.classical(frame)
            rows.append(
                [c["brightness"], c["sharpness"], c["blur"], c["anomaly"], iqa, scene]
            )
        return np.asarray(rows, dtype=np.float32)
