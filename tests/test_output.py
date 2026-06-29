"""Tests for result exporters."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from video_benchmark.output.csv_export import export_video_scores_csv
from video_benchmark.output.json_export import export_detailed_json
from video_benchmark.scoring.scorer import VideoScore


def _score() -> VideoScore:
    return VideoScore(
        operator_id="op_1",
        filename="sample.mp4",
        video_path="/videos/op_1/sample.mp4",
        composite_score=52.0,
        grade="C",
        metric_scores={"brightness": 25.0},
        metric_weights={"brightness": 1.0},
        score_contributions={"brightness": 25.0},
        raw_metrics={"brightness_mean": 20.0},
        worst_issue="poor_lighting",
        recommendations=["Improve lighting: target even workspace illumination."],
        scoring_notes=["Learned IQA unavailable; image quality used fallback."],
    )


def test_json_export_includes_recommendations(tmp_path: Path) -> None:
    path = export_detailed_json(
        [_score()],
        [],
        [],
        tmp_path,
        run_info={"weights_version": "v2", "sample_rate": 1, "model_pyiqa": False},
    )
    data = json.loads(path.read_text())

    assert data["run_info"] == {
        "weights_version": "v2",
        "sample_rate": 1,
        "model_pyiqa": False,
    }
    video = data["video_scores"][0]
    assert video["recommendations"] == [
        "Improve lighting: target even workspace illumination."
    ]
    assert video["scoring_notes"] == [
        "Learned IQA unavailable; image quality used fallback."
    ]
    assert video["metric_weights"] == {"brightness": 1.0}
    assert video["score_contributions"] == {"brightness": 25.0}


def test_video_csv_export_includes_recommendations(tmp_path: Path) -> None:
    path = export_video_scores_csv([_score()], tmp_path)

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["recommendations"] == (
        "Improve lighting: target even workspace illumination."
    )
    assert rows[0]["scoring_notes"] == (
        "Learned IQA unavailable; image quality used fallback."
    )
    assert rows[0]["weight_brightness"] == "1.0"
    assert rows[0]["contribution_brightness"] == "25.0"
