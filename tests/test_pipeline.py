"""Tests for pipeline components."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_benchmark.acceleration import AccelerationInfo, detect_acceleration
from video_benchmark.config import BenchmarkSettings, ScoringWeights, SegmentSpec
from video_benchmark.pipeline.orchestrator import _resolve_worker_count, run_pipeline
from video_benchmark.pipeline.segment_extractor import effective_segments
from video_benchmark.sources.base import VideoFile


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
    def test_default_weights_sum_to_100(self) -> None:
        w = ScoringWeights()
        total = sum(w.as_dict().values())
        assert abs(total - 1.0) < 0.01

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

    def test_segments_clip_to_video_duration(self) -> None:
        segments = [
            SegmentSpec(start_sec=120, end_sec=240),
            SegmentSpec(start_sec=300, end_sec=420),
        ]
        specs = effective_segments(segments, duration_sec=180)
        assert specs == [SegmentSpec(start_sec=120, end_sec=180)]

    def test_segments_fallback_for_short_clip(self) -> None:
        segments = [
            SegmentSpec(start_sec=120, end_sec=240),
            SegmentSpec(start_sec=1680, end_sec=1800),
            SegmentSpec(start_sec=3300, end_sec=3420),
        ]
        specs = effective_segments(segments, duration_sec=45.2)
        assert specs == [SegmentSpec(start_sec=0, end_sec=46)]


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


class TestPipelineIntegration:
    def test_pipeline_scores_short_real_video(self, tmp_path: Path) -> None:
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg is required for pipeline integration test")

        video_path = tmp_path / "operator_1" / "sample.mp4"
        video_path.parent.mkdir()
        _write_test_video(video_path)

        result = run_pipeline(
            [
                VideoFile(
                    operator_id="operator_1",
                    video_path=str(video_path),
                    filename=video_path.name,
                )
            ],
            BenchmarkSettings(
                workers=1,
                sample_rate=2,
                no_gpu=True,
                weights_version="v1",
            ),
        )

        assert result.failed == []
        assert len(result.scores) == 1
        score = result.scores[0]
        assert score.filename == "sample.mp4"
        assert 0.0 <= score.composite_score <= 100.0
        assert score.raw_metrics["total_frames"] > 0
        assert score.segment_scores
        assert result.operator_rankings[0]["operator_id"] == "operator_1"
        assert result.run_info["weights_version"] == "v1"
        assert result.run_info["sample_rate"] == 2


def _write_test_video(path: Path) -> None:
    width = 96
    height = 72
    fps = 6.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV could not create mp4 test video")

    try:
        for idx in range(12):
            frame = np.full((height, width, 3), 96, dtype=np.uint8)
            cv2.rectangle(
                frame,
                (10 + idx, 12),
                (42 + idx, 44),
                (220, 220, 220),
                thickness=-1,
            )
            cv2.putText(
                frame,
                str(idx),
                (54, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
