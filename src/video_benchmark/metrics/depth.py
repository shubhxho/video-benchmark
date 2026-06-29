"""Monocular depth structure metric using Depth Anything V2 (transformers).

Depth Anything V2 (NeurIPS 2024, DINOv2 backbone) produces dense relative-depth
maps. For head-mounted operator footage we use it as a *workspace-structure*
signal: valid task footage has clear near-field structure (hands/tools close to
the camera with depth variation), whereas a frame pointed at a flat wall,
ceiling, or floor is depth-flat and uninformative.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEPTH_AVAILABLE = False
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    _DEPTH_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    return _DEPTH_AVAILABLE


class DepthStructureMetric:
    """Workspace depth-structure scoring via Depth Anything V2.

    Returns a 0-100 score where higher means a richer near-field 3D structure
    (hands/tools present) and lower means a flat, far, or featureless view.
    """

    def __init__(
        self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"
    ) -> None:
        self.model_name = model_name
        self._pipe: Any | None = None
        self._failed = False

    def _ensure_model(self) -> bool:
        if self._pipe is not None:
            return True
        if self._failed or not _DEPTH_AVAILABLE:
            return False
        try:
            import torch
            from transformers import pipeline

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._pipe = pipeline(
                task="depth-estimation",
                model=self.model_name,
                device=device,
            )
            logger.info(f"Loaded Depth Anything model: {self.model_name}")
            return True
        except Exception:
            logger.exception("Failed to load Depth Anything model")
            self._failed = True
            self._pipe = None
            return False

    def compute_structure_score(self, frame: np.ndarray) -> float | None:
        """Score a single BGR frame for workspace depth structure (0-100)."""
        if not self._ensure_model():
            return None

        try:
            import cv2
            from PIL import Image

            assert self._pipe is not None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            result = self._pipe(pil_img)
            depth = result.get("predicted_depth")
            if depth is None:
                # Fall back to the rendered PIL depth map.
                depth_arr = np.asarray(result["depth"], dtype=np.float32)
            else:
                depth_arr = depth.squeeze().detach().cpu().numpy().astype(np.float32)

            return self._score_from_depth(depth_arr)
        except Exception:
            logger.debug("Depth compute failed for frame", exc_info=True)
            return None

    @staticmethod
    def _score_from_depth(depth: np.ndarray) -> float:
        """Convert a relative-depth map into a 0-100 structure score.

        Depth Anything outputs larger values for *nearer* surfaces. We reward
        (a) depth spread — the scene has both near and far content — and
        (b) presence of near-field pixels (something close to the operator).
        """
        d = depth.astype(np.float32)
        d_min, d_max = float(d.min()), float(d.max())
        if d_max - d_min < 1e-6:
            return 0.0  # perfectly flat depth → wall/ceiling/floor

        norm = (d - d_min) / (d_max - d_min)  # 0 (far) .. 1 (near)

        # Spread: normalized std rewards genuine 3D layout over flat planes.
        spread = float(np.std(norm))
        spread_score = min(1.0, spread / 0.25)

        # Near-field presence: fraction of pixels in the closest quartile.
        near_ratio = float(np.mean(norm > 0.75))
        near_score = min(1.0, near_ratio / 0.20)

        score = 0.6 * spread_score + 0.4 * near_score
        return max(0.0, min(100.0, score * 100.0))
