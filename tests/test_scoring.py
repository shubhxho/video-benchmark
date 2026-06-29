"""Tests for scoring, grading, and aggregation."""

from __future__ import annotations

import pytest

from video_benchmark.config import BenchmarkSettings
from video_benchmark.pipeline.orchestrator import VideoMetrics
from video_benchmark.scoring.aggregator import aggregate_operators
from video_benchmark.scoring.grader import assign_grade, grade_description
from video_benchmark.scoring.scorer import (
    VideoScore,
    _build_recommendations,
    _normalize_brightness,
    _normalize_hand_detection_rate,
    _normalize_hand_landmark_quality,
    _normalize_sharpness,
    _normalize_stability,
    _weighted_contributions,
    _weighted_score,
    score_video,
)
from video_benchmark.sources.base import VideoFile


class TestGrader:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (95, "A"),
            (80, "A"),
            (79, "B"),
            (60, "B"),
            (59, "C"),
            (40, "C"),
            (39, "D"),
            (20, "D"),
            (19, "F"),
            (0, "F"),
        ],
    )
    def test_assign_grade(self, score: float, expected: str) -> None:
        assert assign_grade(score) == expected

    def test_grade_description(self) -> None:
        assert "Excellent" in grade_description("A")
        assert "Unusable" in grade_description("F")


class TestNormalization:
    def test_brightness_optimal(self) -> None:
        score = _normalize_brightness([130.0, 140.0, 120.0])
        assert score > 70

    def test_brightness_dark(self) -> None:
        score = _normalize_brightness([10.0, 15.0, 20.0])
        assert score < 30

    def test_sharpness_high(self) -> None:
        score = _normalize_sharpness([500.0, 600.0])
        assert score >= 100.0

    def test_sharpness_low(self) -> None:
        score = _normalize_sharpness([10.0, 20.0])
        assert score < 10

    def test_stability_perfect(self) -> None:
        score = _normalize_stability([0.0, 0.0, 0.0])
        assert score == 100.0

    def test_stability_shaky(self) -> None:
        score = _normalize_stability([25.0, 30.0])
        assert score < 10

    def test_hand_detection_rate(self) -> None:
        assert _normalize_hand_detection_rate(1.0) == 100.0
        assert _normalize_hand_detection_rate(0.0) == 0.0
        assert _normalize_hand_detection_rate(0.5) == 50.0

    def test_landmark_quality_full(self) -> None:
        score = _normalize_hand_landmark_quality([21, 21, 21])
        assert score >= 75

    def test_landmark_quality_empty(self) -> None:
        assert _normalize_hand_landmark_quality([]) == 0.0

    def test_weighted_score_normalizes_custom_weights(self) -> None:
        score = _weighted_score({"a": 100.0, "b": 50.0}, {"a": 2.0, "b": 2.0})
        assert score == 75.0

    def test_weighted_score_ignores_negative_weights(self) -> None:
        score = _weighted_score({"a": 100.0, "b": 0.0}, {"a": 1.0, "b": -5.0})
        assert score == 100.0

    def test_weighted_contributions_sum_to_score(self) -> None:
        contributions = _weighted_contributions(
            {"a": 100.0, "b": 50.0},
            {"a": 2.0, "b": 2.0},
        )

        assert contributions == {"a": 50.0, "b": 25.0}
        assert sum(contributions.values()) == 75.0

    def test_recommendations_prioritize_weakest_known_metrics(self) -> None:
        recommendations = _build_recommendations(
            {
                "stability": 45.0,
                "brightness": 20.0,
                "unknown_metric": 0.0,
                "sharpness": 80.0,
                "hand_detection_rate": 55.0,
            }
        )
        assert len(recommendations) == 3
        assert recommendations[0].startswith("Improve lighting")
        assert recommendations[1].startswith("Reduce camera shake")
        assert recommendations[2].startswith("Improve hand visibility")

    def test_recommendations_empty_for_good_scores(self) -> None:
        assert _build_recommendations({"brightness": 90.0, "stability": 85.0}) == []


class TestVideoScoring:
    def test_v2_scoring_notes_explain_fallbacks(self) -> None:
        video = VideoFile(
            operator_id="op_1",
            filename="sample.mp4",
            video_path="/videos/op_1/sample.mp4",
        )
        metrics = VideoMetrics(
            brightness=[100.0, 120.0],
            sharpness=[100.0, 120.0],
            stability=[1.0],
            hand_detection_rate=1.0,
            hand_landmark_counts=[21, 21],
            tracking_continuity=100.0,
            anomaly_scores=[100.0],
            blur_scores=[100.0],
            temporal_consistency=100.0,
        )

        score = score_video(video, metrics, BenchmarkSettings(weights_version="v2"))

        assert any("Learned IQA unavailable" in note for note in score.scoring_notes)
        assert any("Scene validation unavailable" in note for note in score.scoring_notes)
        assert any("Audio analysis unavailable" in note for note in score.scoring_notes)
        assert score.metric_weights
        assert score.score_contributions
        assert abs(sum(score.score_contributions.values()) - score.composite_score) < 0.1

    def test_v2_scoring_notes_empty_when_optional_signals_exist(self) -> None:
        video = VideoFile(
            operator_id="op_1",
            filename="sample.mp4",
            video_path="/videos/op_1/sample.mp4",
        )
        metrics = VideoMetrics(
            brightness=[100.0],
            sharpness=[100.0],
            stability=[1.0],
            hand_detection_rate=1.0,
            hand_landmark_counts=[21],
            tracking_continuity=100.0,
            iqa_scores=[85.0],
            scene_validity_scores=[90.0],
            anomaly_scores=[100.0],
            blur_scores=[100.0],
            temporal_consistency=100.0,
            audio_quality=80.0,
            audio_details={"snr_db": 30.0},
            vqa_overall=[88.0],
            depth_structure_scores=[80.0],
        )

        score = score_video(video, metrics, BenchmarkSettings(weights_version="v2"))

        assert score.scoring_notes == []


class TestAggregator:
    def _make_score(self, op_id: str, composite: float) -> VideoScore:
        return VideoScore(
            operator_id=op_id,
            filename=f"{op_id}_video.mp4",
            video_path=f"/videos/{op_id}/video.mp4",
            composite_score=composite,
            grade=assign_grade(composite),
            metric_scores={},
            raw_metrics={},
        )

    def test_single_operator(self) -> None:
        scores = [self._make_score("op_1", 85.0)]
        rankings = aggregate_operators(scores)
        assert len(rankings) == 1
        assert rankings[0]["operator_id"] == "op_1"
        assert rankings[0]["rank"] == 1

    def test_multiple_operators_ranked(self) -> None:
        scores = [
            self._make_score("op_bad", 30.0),
            self._make_score("op_good", 90.0),
            self._make_score("op_mid", 60.0),
        ]
        rankings = aggregate_operators(scores)
        assert rankings[0]["operator_id"] == "op_good"
        assert rankings[-1]["operator_id"] == "op_bad"

    def test_consistency_bonus(self) -> None:
        # Consistent operator gets a bonus
        scores = [
            self._make_score("op_c", 80.0),
            self._make_score("op_c", 81.0),
            self._make_score("op_c", 79.0),
        ]
        rankings = aggregate_operators(scores)
        assert rankings[0]["consistency_bonus"] > 0

    def test_inconsistent_no_bonus(self) -> None:
        scores = [
            self._make_score("op_i", 90.0),
            self._make_score("op_i", 30.0),
        ]
        rankings = aggregate_operators(scores)
        assert rankings[0]["consistency_bonus"] == 0.0
