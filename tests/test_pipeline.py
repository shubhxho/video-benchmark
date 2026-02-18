"""Tests for pipeline components."""

from __future__ import annotations

from video_benchmark.acceleration import AccelerationInfo, detect_acceleration
from video_benchmark.config import BenchmarkSettings, ScoringWeights
from video_benchmark.pipeline.orchestrator import _resolve_worker_count


class TestAcceleration:
    def test_detect_returns_info(self) -> None:
        info = detect_acceleration()
        assert isinstance(info, AccelerationInfo)

    def test_no_gpu_disables_hwaccel(self) -> None:
        info = detect_acceleration(force_no_gpu=True)
        assert not info.videotoolbox
        assert info.hwaccel_args == []

    def test_hwaccel_args_empty_without_videotoolbox(self) -> None:
        info = AccelerationInfo(videotoolbox=False)
        assert info.hwaccel_args == []

    def test_hwaccel_args_set_with_videotoolbox(self) -> None:
        info = AccelerationInfo(videotoolbox=True)
        assert info.hwaccel_args == ["-hwaccel", "videotoolbox"]


class TestConfig:
    def test_default_weights_sum_to_110(self) -> None:
        w = ScoringWeights()
        total = sum(w.as_dict().values())
        assert abs(total - 1.10) < 0.01

    def test_default_segments(self) -> None:
        s = BenchmarkSettings()
        specs = s.segment_specs()
        assert len(specs) == 3
        assert specs[0].start_sec == 120
        assert specs[0].end_sec == 240

    def test_custom_segment_count(self) -> None:
        s = BenchmarkSettings(segments=2)
        specs = s.segment_specs()
        assert len(specs) == 2

    def test_weights_from_dict(self) -> None:
        w = ScoringWeights(brightness=0.5, sharpness=0.5)
        assert w.brightness == 0.5


class TestWorkerResolution:
    def test_respects_explicit_worker_count(self) -> None:
        accel = AccelerationInfo(videotoolbox=True, mps_available=True)
        workers = _resolve_worker_count(3, total_videos=10, accel=accel)
        assert workers == 3

    def test_auto_workers_on_gpu_path(self) -> None:
        accel = AccelerationInfo(videotoolbox=True, mps_available=True)
        workers = _resolve_worker_count(0, total_videos=6, accel=accel)
        assert 2 <= workers <= 6

    def test_single_video_always_single_worker(self) -> None:
        accel = AccelerationInfo(videotoolbox=True, mps_available=True)
        workers = _resolve_worker_count(0, total_videos=1, accel=accel)
        assert workers == 1
