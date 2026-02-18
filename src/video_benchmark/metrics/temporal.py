"""Temporal consistency metric — quality drops, flickering, duplicate frames."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TemporalResult:
    consistency_score: float  # 0-100
    quality_drops: list[int] = field(default_factory=list)
    flicker_score: float = 100.0  # 100 = no flicker
    duplicate_frames: list[int] = field(default_factory=list)


class TemporalConsistencyMetric:
    """Analyze temporal consistency across a sequence of frames."""

    def compute(self, frames: list[np.ndarray]) -> TemporalResult:
        """Compute temporal consistency metrics for a frame sequence."""
        if len(frames) < 2:
            return TemporalResult(consistency_score=100.0)

        quality_drops = self._detect_quality_drops(frames)
        flicker_score = self._detect_flicker(frames)
        duplicate_frames = self._detect_duplicates(frames)

        # Overall score: penalize for issues
        n = len(frames)
        drop_penalty = min(50.0, len(quality_drops) / n * 200)
        flicker_penalty = max(0.0, (100.0 - flicker_score) * 0.3)
        dupe_penalty = min(30.0, len(duplicate_frames) / n * 100)

        consistency = max(
            0.0, 100.0 - drop_penalty - flicker_penalty - dupe_penalty
        )

        return TemporalResult(
            consistency_score=consistency,
            quality_drops=quality_drops,
            flicker_score=flicker_score,
            duplicate_frames=duplicate_frames,
        )

    def _detect_quality_drops(self, frames: list[np.ndarray]) -> list[int]:
        """Find frame indices where sharpness drops > 2σ below rolling mean."""
        variances = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            variances.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        if len(variances) < 5:
            return []

        arr = np.array(variances)
        # Use a rolling window of 5 frames
        window = 5
        drops: list[int] = []
        for i in range(window, len(arr)):
            window_slice = arr[max(0, i - window) : i]
            mean = float(np.mean(window_slice))
            std = float(np.std(window_slice))
            if std > 0 and arr[i] < mean - 2 * std:
                drops.append(i)

        return drops

    def _detect_flicker(self, frames: list[np.ndarray]) -> float:
        """Detect flickering via frame-to-frame intensity changes.

        Returns 0-100 score where 100 = no flicker.
        """
        intensities = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            intensities.append(float(np.mean(gray)))

        if len(intensities) < 2:
            return 100.0

        # Frame-to-frame intensity differences
        diffs = [
            abs(intensities[i] - intensities[i - 1])
            for i in range(1, len(intensities))
        ]
        mean_diff = sum(diffs) / len(diffs)

        # Count large intensity swings (>15 intensity levels)
        large_swings = sum(1 for d in diffs if d > 15)
        swing_ratio = large_swings / len(diffs)

        # Score: penalize high mean diff and many large swings
        penalty = min(50.0, mean_diff * 2) + min(50.0, swing_ratio * 200)
        return max(0.0, 100.0 - penalty)

    def _detect_duplicates(self, frames: list[np.ndarray]) -> list[int]:
        """Find duplicate/frozen frames using structural similarity.

        SSIM > 0.99 between consecutive frames = likely duplicate.
        Uses a fast approximation via mean absolute difference.
        """
        dupes: list[int] = []
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Fast SSIM approximation: normalized cross-correlation
            diff = cv2.absdiff(prev_gray, curr_gray)
            mean_diff = float(np.mean(diff))

            # Very low difference = essentially the same frame
            if mean_diff < 0.5:
                dupes.append(i)

        return dupes
