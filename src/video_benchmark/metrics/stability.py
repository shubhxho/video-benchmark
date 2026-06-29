"""Camera stability metric — Farneback optical flow between consecutive frames."""

from __future__ import annotations

import cv2
import numpy as np

from video_benchmark.metrics.base import Metric


class StabilityMetric(Metric):
    name = "stability"

    def compute(self, frame: np.ndarray) -> float:
        raise NotImplementedError("Use compute_flow() with two frames instead.")

    def compute_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Compute mean optical flow magnitude between two frames.

        Lower values = more stable camera. Returns mean flow magnitude.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        initial_flow = np.zeros((*prev_gray.shape, 2), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            initial_flow,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))
