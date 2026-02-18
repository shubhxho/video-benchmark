"""Learned image quality assessment using pyiqa (TOPIQ, MUSIQ, BRISQUE)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_PYIQA_AVAILABLE = False
try:
    import pyiqa
    import torch

    _PYIQA_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    return _PYIQA_AVAILABLE


class LearnedIQAMetric:
    """No-reference image quality using learned models.

    Supported models: topiq_nr, musiq, brisque, clipiqa, niqe
    """

    def __init__(self, model_name: str = "topiq_nr") -> None:
        self.model_name = model_name
        self._model = None
        self._device = None

    def _ensure_model(self) -> bool:
        """Lazily initialize the pyiqa model on first use."""
        if self._model is not None:
            return True
        if not _PYIQA_AVAILABLE:
            logger.warning("pyiqa not installed — IQA metric unavailable")
            return False
        try:
            self._device = torch.device("cpu")
            self._model = pyiqa.create_metric(
                self.model_name, device=self._device
            )
            logger.info(f"Loaded IQA model: {self.model_name}")
            return True
        except Exception:
            logger.exception(f"Failed to load IQA model: {self.model_name}")
            self._model = None
            return False

    def compute(self, frame: np.ndarray) -> float | None:
        """Score a single BGR frame. Returns 0-100 or None if unavailable.

        pyiqa models typically output scores where higher = better quality.
        We normalize to 0-100 range.
        """
        if not self._ensure_model():
            return None

        try:
            import cv2

            # BGR → RGB → float32 tensor [0, 1] → [1, C, H, W]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            tensor = tensor.to(self._device)

            with torch.no_grad():
                raw_score = self._model(tensor).item()

            return self._normalize(raw_score)
        except Exception:
            logger.debug("IQA compute failed for frame", exc_info=True)
            return None

    def _normalize(self, raw_score: float) -> float:
        """Normalize model-specific raw score to 0-100."""
        # pyiqa models with lower_better=False: higher is better
        # topiq_nr: range ~0.0-1.0
        # musiq: range ~0-100
        # brisque: range 0-100, lower is better
        if self._model is not None and hasattr(self._model, "lower_better"):
            if self._model.lower_better:
                # Lower raw = better quality (e.g., BRISQUE, NIQE)
                # Typical BRISQUE range: 0 (best) - 100 (worst)
                return max(0.0, min(100.0, 100.0 - raw_score))

        # Higher raw = better quality (e.g., TOPIQ, MUSIQ, CLIPIQA)
        if self.model_name in ("topiq_nr", "clipiqa", "clipiqa+"):
            # Range ~0.0-1.0 → 0-100
            return max(0.0, min(100.0, raw_score * 100.0))
        if self.model_name == "musiq":
            # Range ~0-100
            return max(0.0, min(100.0, raw_score))

        # Default: assume 0-1 range
        return max(0.0, min(100.0, raw_score * 100.0))
