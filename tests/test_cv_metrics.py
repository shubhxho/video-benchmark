"""Unit tests for the cutting-edge CV metric scoring logic.

These cover the deterministic, model-free parts (no weights downloaded):
- Depth Anything V2 structure scoring from a depth map.
- DOVER technical/aesthetic fusion.
"""

from __future__ import annotations

import numpy as np

from video_benchmark.metrics.depth import DepthStructureMetric
from video_benchmark.metrics.vqa import DOVERVideoQualityMetric


class TestDepthStructureScore:
    def test_flat_depth_scores_zero(self) -> None:
        flat = np.ones((32, 32), dtype=np.float32)
        assert DepthStructureMetric._score_from_depth(flat) == 0.0

    def test_structured_depth_scores_high(self) -> None:
        gradient = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
        score = DepthStructureMetric._score_from_depth(gradient)
        assert score > 50.0

    def test_score_is_bounded(self) -> None:
        rng = np.random.default_rng(0)
        noise = rng.random((48, 48)).astype(np.float32)
        score = DepthStructureMetric._score_from_depth(noise)
        assert 0.0 <= score <= 100.0


class TestDOVERFusion:
    def test_fuse_is_bounded(self) -> None:
        assert 0.0 <= DOVERVideoQualityMetric._fuse(0.0, 0.0) <= 1.0

    def test_fuse_monotonic_in_technical(self) -> None:
        low = DOVERVideoQualityMetric._fuse(0.0, 0.0)
        high = DOVERVideoQualityMetric._fuse(0.3, 0.0)
        assert high > low

    def test_fuse_monotonic_in_aesthetic(self) -> None:
        low = DOVERVideoQualityMetric._fuse(0.1, -0.1)
        high = DOVERVideoQualityMetric._fuse(0.1, 0.1)
        assert high > low
