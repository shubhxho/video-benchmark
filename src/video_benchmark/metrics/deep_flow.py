"""Deep optical flow stability using RAFT (torchvision)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_RAFT_AVAILABLE = False
try:
    import torch
    from torchvision.models.optical_flow import (
        Raft_Small_Weights,
        raft_small,
    )

    _RAFT_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    return _RAFT_AVAILABLE


class RAFTStabilityMetric:
    """Camera stability measurement using RAFT deep optical flow.

    Falls back to Farneback if torchvision is not installed.
    """

    def __init__(self) -> None:
        self._model = None
        self._transforms = None
        self._device = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if not _RAFT_AVAILABLE:
            return False
        try:
            weights = Raft_Small_Weights.DEFAULT
            self._transforms = weights.transforms()
            self._model = raft_small(weights=weights, progress=False)
            self._device = torch.device("cpu")
            self._model = self._model.to(self._device).eval()
            logger.info("Loaded RAFT optical flow model")
            return True
        except Exception:
            logger.exception("Failed to load RAFT model")
            self._model = None
            return False

    def compute_flow(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> float | None:
        """Compute mean optical flow magnitude using RAFT.

        Returns mean flow magnitude (pixels), or None if unavailable.
        Lower values = more stable camera.
        """
        if not self._ensure_model():
            return None

        try:
            import cv2

            # Convert BGR → RGB → tensor [C, H, W] uint8
            prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

            prev_t = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0)
            curr_t = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0)

            # RAFT requires specific input size — resize to 520x960
            prev_t = torch.nn.functional.interpolate(
                prev_t.float(), size=(520, 960), mode="bilinear",
                align_corners=False
            ).to(torch.uint8)
            curr_t = torch.nn.functional.interpolate(
                curr_t.float(), size=(520, 960), mode="bilinear",
                align_corners=False
            ).to(torch.uint8)

            # Apply RAFT transforms
            prev_t, curr_t = self._transforms(prev_t, curr_t)
            prev_t = prev_t.to(self._device)
            curr_t = curr_t.to(self._device)

            with torch.no_grad():
                flow_list = self._model(prev_t, curr_t)
                # Last element is the final refined flow
                flow = flow_list[-1]  # [1, 2, H, W]

            # Compute magnitude
            flow_np = flow.squeeze(0).cpu().numpy()  # [2, H, W]
            mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2)
            return float(np.mean(mag))

        except Exception:
            logger.debug("RAFT compute failed", exc_info=True)
            return None
