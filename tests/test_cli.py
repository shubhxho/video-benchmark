"""Tests for CLI behavior."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from video_benchmark.cli import app
from video_benchmark.pipeline.orchestrator import PipelineResult
from video_benchmark.scoring.scorer import VideoScore
from video_benchmark.sources.base import VideoFile

runner = CliRunner()


def _score() -> VideoScore:
    return VideoScore(
        operator_id="videos",
        filename="sample.mp4",
        video_path="/tmp/sample.mp4",
        composite_score=42.0,
        grade="C",
        metric_scores={"brightness": 20.0, "stability": 85.0},
        raw_metrics={"brightness_mean": 18.0},
        worst_issue="poor_lighting",
        recommendations=["Improve lighting: target even workspace illumination."],
    )


def test_score_single_prints_scorecard(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"not a real video; pipeline is mocked")

    monkeypatch.setattr("video_benchmark.cli.require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        "video_benchmark.cli.run_pipeline",
        lambda videos, settings: PipelineResult(
            scores=[_score()],
            failed=[],
            operator_rankings=[],
        ),
    )

    result = runner.invoke(app, ["score-single", str(video)])

    assert result.exit_code == 0
    assert "Recommended Fixes" in result.output
    assert "Improve lighting" in result.output


def test_score_single_exits_nonzero_when_processing_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"not a real video; pipeline is mocked")

    monkeypatch.setattr("video_benchmark.cli.require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        "video_benchmark.cli.run_pipeline",
        lambda videos, settings: PipelineResult(
            scores=[],
            failed=[
                (
                    VideoFile(
                        operator_id="videos",
                        video_path=str(video),
                        filename=video.name,
                    ),
                    "No frames could be extracted",
                )
            ],
            operator_rankings=[],
        ),
    )

    result = runner.invoke(app, ["score-single", str(video)])

    assert result.exit_code == 1
    assert "Failed Videos" in result.output
    assert "No frames could be extracted" in result.output
