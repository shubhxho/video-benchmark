"""Frame anomaly detection — blocked lens, glare, overexposure, corruption."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class AnomalyResult:
    flags: list[str] = field(default_factory=list)
    score: float = 100.0  # 100 = no anomalies


class AnomalyDetector:
    """Detect frame-level anomalies using pure OpenCV heuristics."""

    def detect_anomalies(self, frame: np.ndarray) -> list[str]:
        """Return list of anomaly flags found in the frame."""
        flags: list[str] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # --- Blocked lens: very low variance + very few edges ---
        pixel_var = float(np.var(gray))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (h * w)
        if pixel_var < 100 and edge_density < 0.01:
            flags.append("blocked_lens")

        # --- Underexposure: >80% of pixels below intensity 30 ---
        dark_ratio = float(np.count_nonzero(gray < 30)) / (h * w)
        if dark_ratio > 0.80:
            flags.append("underexposure")

        # --- Overexposure: >20% of pixels clipped at 255 in all channels ---
        clipped = np.all(frame == 255, axis=2)
        clip_ratio = float(np.count_nonzero(clipped)) / (h * w)
        if clip_ratio > 0.20:
            flags.append("overexposure")

        # --- Glare/reflection: low sat + high value in HSV ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        glare_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
        glare_ratio = float(np.count_nonzero(glare_mask)) / (h * w)
        if glare_ratio > 0.15:
            flags.append("glare")

        # --- Frame corruption: solid color or half-black ---
        top_half_mean = float(np.mean(gray[: h // 2, :]))
        bottom_half_mean = float(np.mean(gray[h // 2 :, :]))
        if pixel_var < 5:
            flags.append("solid_color")
        elif abs(top_half_mean - bottom_half_mean) > 100 and (
            top_half_mean < 10 or bottom_half_mean < 10
        ):
            flags.append("half_black")

        # --- Extreme color cast ---
        means = [float(np.mean(frame[:, :, c])) for c in range(3)]
        total_mean = sum(means) / 3.0
        if total_mean > 20:
            ratios = [m / total_mean for m in means]
            if any(r > 1.8 or r < 0.2 for r in ratios):
                flags.append("color_cast")

        return flags

    def compute_anomaly_score(self, frame: np.ndarray) -> float:
        """Return 0-100 score where 100 = no anomalies detected."""
        flags = self.detect_anomalies(frame)
        if not flags:
            return 100.0
        # Each anomaly costs 25 points, minimum score is 0
        penalty = len(flags) * 25.0
        return max(0.0, 100.0 - penalty)
