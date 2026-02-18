"""Blur classification — motion blur, defocus blur, encoding blur."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BlurResult:
    is_blurry: bool
    blur_type: str  # "none", "motion", "defocus", "encoding"
    severity: float  # 0 (sharp) to 100 (extremely blurry)


# Threshold for Laplacian variance below which a frame is considered blurry
BLUR_THRESHOLD = 100.0


class BlurClassifier:
    """Classify blur type using FFT and DCT analysis."""

    def __init__(self, blur_threshold: float = BLUR_THRESHOLD) -> None:
        self.blur_threshold = blur_threshold

    def classify(self, frame: np.ndarray) -> BlurResult:
        """Classify blur type and severity."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if lap_var >= self.blur_threshold:
            return BlurResult(is_blurry=False, blur_type="none", severity=0.0)

        # Severity: 0 at threshold, 100 at lap_var=0
        severity = min(100.0, (1.0 - lap_var / self.blur_threshold) * 100.0)

        # Classify blur type via FFT
        blur_type = self._classify_type(gray)

        return BlurResult(is_blurry=True, blur_type=blur_type, severity=severity)

    def _classify_type(self, gray: np.ndarray) -> str:
        """Use FFT analysis to distinguish motion vs defocus blur."""
        # Compute magnitude spectrum
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # --- Motion blur: directional energy in frequency domain ---
        # Compute angular energy histogram
        motion_score = self._detect_motion_blur(magnitude, cy, cx)

        # --- Defocus blur: ring pattern / radial attenuation ---
        defocus_score = self._detect_defocus_blur(magnitude, cy, cx)

        # --- Encoding blur: blockiness from DCT ---
        encoding_score = self._detect_encoding_blur(gray)

        scores = {
            "motion": motion_score,
            "defocus": defocus_score,
            "encoding": encoding_score,
        }
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _detect_motion_blur(
        self, magnitude: np.ndarray, cy: int, cx: int
    ) -> float:
        """Motion blur creates directional streaks in frequency domain.

        Measure angular energy concentration: high concentration = motion blur.
        """
        h, w = magnitude.shape
        n_bins = 36
        angular_energy = np.zeros(n_bins)

        # Sample points in the frequency domain
        max_r = min(cy, cx) // 2
        for r in range(10, max_r, 2):
            for bin_idx in range(n_bins):
                angle = 2 * np.pi * bin_idx / n_bins
                y = int(cy + r * np.sin(angle))
                x = int(cx + r * np.cos(angle))
                if 0 <= y < h and 0 <= x < w:
                    angular_energy[bin_idx] += magnitude[y, x]

        if np.sum(angular_energy) == 0:
            return 0.0

        # Normalize to distribution
        angular_energy /= np.sum(angular_energy)
        # High concentration = low entropy = motion blur
        entropy = -np.sum(
            angular_energy * np.log2(angular_energy + 1e-10)
        )
        max_entropy = np.log2(n_bins)
        # Low entropy → high motion score
        return max(0.0, (1.0 - entropy / max_entropy) * 100.0)

    def _detect_defocus_blur(
        self, magnitude: np.ndarray, cy: int, cx: int
    ) -> float:
        """Defocus blur attenuates high frequencies radially.

        Measure ratio of high-freq to low-freq energy.
        """
        h, w = magnitude.shape
        max_r = min(cy, cx)

        # Create radial distance map
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

        # Low freq energy (inner 30%)
        low_mask = dist < max_r * 0.3
        low_energy = float(np.sum(magnitude[low_mask]))

        # High freq energy (outer 50%)
        high_mask = dist > max_r * 0.5
        high_energy = float(np.sum(magnitude[high_mask]))

        if low_energy == 0:
            return 0.0

        # Strong radial attenuation = defocus
        ratio = high_energy / low_energy
        # Lower ratio = more defocus. Typical sharp image ratio ~0.5-1.0
        return max(0.0, min(100.0, (1.0 - ratio) * 100.0))

    def _detect_encoding_blur(self, gray: np.ndarray) -> float:
        """Detect blockiness from JPEG/H.264 encoding artifacts.

        Measure energy at 8x8 block boundaries using DCT analysis.
        """
        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.0

        # Measure horizontal and vertical edge strength at 8-pixel intervals
        boundary_energy = 0.0
        non_boundary_energy = 0.0
        count_b = 0
        count_nb = 0

        for y in range(1, h - 1):
            diff = float(
                np.mean(np.abs(
                    gray[y, :].astype(np.float32) - gray[y - 1, :].astype(np.float32)
                ))
            )
            if y % 8 == 0:
                boundary_energy += diff
                count_b += 1
            else:
                non_boundary_energy += diff
                count_nb += 1

        if count_b == 0 or count_nb == 0:
            return 0.0

        avg_boundary = boundary_energy / count_b
        avg_non_boundary = non_boundary_energy / count_nb

        if avg_non_boundary == 0:
            return 0.0

        # High boundary-to-non-boundary ratio = blockiness
        ratio = avg_boundary / avg_non_boundary
        # Ratio > 1.5 indicates strong blockiness
        return max(0.0, min(100.0, (ratio - 1.0) * 100.0))
