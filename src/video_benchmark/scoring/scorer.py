"""Composite scoring engine — combine metric scores into a single video score."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from video_benchmark.config import BenchmarkSettings
from video_benchmark.metrics.brightness import BrightnessMetric
from video_benchmark.scoring.grader import assign_grade
from video_benchmark.sources.base import VideoFile

if TYPE_CHECKING:
    from video_benchmark.pipeline.orchestrator import VideoMetrics

type MetricValue = float | bool
type SegmentScore = dict[str, float | int]


@dataclass
class VideoScore:
    operator_id: str
    filename: str
    video_path: str
    composite_score: float
    grade: str
    metric_scores: dict[str, float]
    raw_metrics: dict[str, MetricValue]
    metric_weights: dict[str, float] = field(default_factory=dict)
    score_contributions: dict[str, float] = field(default_factory=dict)
    segment_scores: list[SegmentScore] = field(default_factory=list)
    worst_issue: str = "none"
    recommendations: list[str] = field(default_factory=list)
    scoring_notes: list[str] = field(default_factory=list)


def _mean(vals: Sequence[float | int]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _weighted_score(metric_scores: dict[str, float], weights: dict[str, float]) -> float:
    """Compute a bounded weighted score, normalizing custom weight files."""
    return sum(_weighted_contributions(metric_scores, weights).values())


def _weighted_contributions(
    metric_scores: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    """Return per-metric point contributions using normalized non-negative weights."""
    total_weight = sum(max(0.0, weight) for weight in weights.values())
    if total_weight <= 0:
        return {metric: 0.0 for metric in weights}
    contributions: dict[str, float] = {}
    for metric, weight in weights.items():
        normalized_weight = max(0.0, weight) / total_weight
        metric_score = max(0.0, min(100.0, metric_scores.get(metric, 0.0)))
        contributions[metric] = metric_score * normalized_weight
    return contributions


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    total_weight = sum(max(0.0, weight) for weight in weights.values())
    if total_weight <= 0:
        return {metric: 0.0 for metric in weights}
    return {metric: max(0.0, weight) / total_weight for metric, weight in weights.items()}


def _normalize_brightness(values: list[float]) -> float:
    """Normalize brightness values to 0-100 score."""
    if not values:
        return 0.0
    scores = [BrightnessMetric.normalize(v) for v in values]
    return sum(scores) / len(scores)


def _normalize_sharpness(values: list[float]) -> float:
    """Normalize Laplacian variance to 0-100 score."""
    if not values:
        return 0.0
    mean_val = sum(values) / len(values)
    return min(100.0, mean_val / 5.0)


def _normalize_stability(values: list[float]) -> float:
    """Normalize optical flow to 0-100 score. Lower flow = higher score."""
    if not values:
        return 50.0
    mean_flow = sum(values) / len(values)
    return max(0.0, min(100.0, 100.0 - mean_flow * 5.0))


def _normalize_hand_detection_rate(rate: float) -> float:
    return rate * 100.0


def _normalize_hand_landmark_quality(counts: list[int]) -> float:
    if not counts:
        return 0.0
    mean_count = sum(counts) / len(counts)
    return min(100.0, mean_count / 21.0 * 80.0)


ISSUE_NAMES = {
    "brightness": "poor_lighting",
    "sharpness": "blurry_frames",
    "stability": "camera_shake",
    "hand_detection_rate": "hands_not_visible",
    "hand_landmark_quality": "poor_hand_visibility",
    "tracking_continuity": "frequent_dropouts",
    "image_quality": "poor_image_quality",
    "video_quality": "poor_video_quality",
    "depth_structure": "flat_or_featureless_view",
    "scene_validity": "wrong_camera_angle",
    "anomaly_score": "frame_anomalies",
    "blur_score": "excessive_blur",
    "temporal_consistency": "quality_inconsistent",
    "audio_quality": "poor_audio",
}

RECOMMENDATIONS = {
    "brightness": (
        "Improve lighting: target even workspace illumination and avoid dark "
        "head-mounted footage."
    ),
    "sharpness": (
        "Improve focus/sharpness: clean the lens, check focus, and reduce "
        "compression before analysis."
    ),
    "stability": (
        "Reduce camera shake: tighten the mount and avoid head movements "
        "during critical actions."
    ),
    "hand_detection_rate": (
        "Improve hand visibility: keep both hands inside the camera view "
        "during task steps."
    ),
    "hand_landmark_quality": (
        "Improve hand detail: move hands closer to the action zone and avoid "
        "occlusion."
    ),
    "tracking_continuity": (
        "Reduce tracking dropouts: keep hands continuously visible through "
        "the full segment."
    ),
    "image_quality": (
        "Improve image quality: fix exposure, focus, lens cleanliness, and "
        "source compression."
    ),
    "video_quality": (
        "Improve overall video quality: reduce compression artifacts, motion "
        "blur, and exposure swings across the clip (DOVER technical/aesthetic)."
    ),
    "depth_structure": (
        "Frame the work surface: keep hands and tools in the near field rather "
        "than pointing at a flat wall, ceiling, or distant area."
    ),
    "scene_validity": "Check camera angle: keep the workspace, tools, and operator hands in frame.",
    "anomaly_score": (
        "Remove frame anomalies: check for blocked lens, glare, overexposure, "
        "corruption, or color cast."
    ),
    "blur_score": (
        "Reduce blur: improve focus, increase light, and stabilize fast head "
        "or hand motion."
    ),
    "temporal_consistency": (
        "Improve temporal consistency: avoid flicker, duplicated frames, and "
        "sudden quality drops."
    ),
    "audio_quality": (
        "Improve audio: reduce wind/noise, avoid long silence, and keep speech "
        "or task audio audible."
    ),
}


def _identify_worst_issue(metric_scores: dict[str, float]) -> str:
    if not metric_scores:
        return "none"
    worst = min(metric_scores, key=lambda k: metric_scores[k])
    if metric_scores[worst] < 40:
        return ISSUE_NAMES.get(worst, worst)
    return "none"


def _build_recommendations(metric_scores: dict[str, float], limit: int = 3) -> list[str]:
    """Return prioritized human actions for the weakest metrics."""
    weak_metrics = [
        (metric, score)
        for metric, score in sorted(metric_scores.items(), key=lambda item: item[1])
        if score < 70.0 and metric in RECOMMENDATIONS
    ]
    return [RECOMMENDATIONS[metric] for metric, _ in weak_metrics[:limit]]


def _score_v1(
    video: VideoFile,
    metrics: VideoMetrics,
    settings: BenchmarkSettings,
) -> VideoScore:
    """V1 scoring: classical CV metrics only."""
    w = settings.weights
    metric_scores = {
        "brightness": _normalize_brightness(metrics.brightness),
        "sharpness": _normalize_sharpness(metrics.sharpness),
        "stability": _normalize_stability(metrics.stability),
        "hand_detection_rate": _normalize_hand_detection_rate(
            metrics.hand_detection_rate
        ),
        "hand_landmark_quality": _normalize_hand_landmark_quality(
            metrics.hand_landmark_counts
        ),
        "tracking_continuity": metrics.tracking_continuity,
    }

    raw_metrics = {
        "brightness_mean": _mean(metrics.brightness),
        "sharpness_mean": _mean(metrics.sharpness),
        "stability_mean": _mean(metrics.stability),
        "hand_detection_rate": metrics.hand_detection_rate,
        "hand_confidence_mean": _mean(metrics.hand_confidence),
        "landmark_count_mean": _mean(metrics.hand_landmark_counts),
        "tracking_continuity": metrics.tracking_continuity,
        "total_frames": len(metrics.brightness),
    }

    wd = w.as_dict()
    contributions = _weighted_contributions(metric_scores, wd)
    composite = max(0.0, min(100.0, sum(contributions.values())))

    return VideoScore(
        operator_id=video.operator_id,
        filename=video.filename,
        video_path=video.video_path,
        composite_score=round(composite, 1),
        grade=assign_grade(composite),
        metric_scores={k: round(v, 1) for k, v in metric_scores.items()},
        metric_weights={k: round(v, 4) for k, v in _normalized_weights(wd).items()},
        score_contributions={k: round(v, 2) for k, v in contributions.items()},
        raw_metrics={k: round(v, 2) for k, v in raw_metrics.items()},
        segment_scores=metrics.segment_scores,
        worst_issue=_identify_worst_issue(metric_scores),
        recommendations=_build_recommendations(metric_scores),
        scoring_notes=["v1 classical CV scoring; optional ML metrics were not used."],
    )


def _score_v2(
    video: VideoFile,
    metrics: VideoMetrics,
    settings: BenchmarkSettings,
) -> VideoScore:
    """V2 scoring: ML models + new metrics."""
    w = settings.weights_v2

    # Image quality: use IQA if available, fallback to brightness+sharpness avg
    scoring_notes: list[str] = []
    if metrics.iqa_scores:
        image_quality = _mean(metrics.iqa_scores)
    else:
        bri = _normalize_brightness(metrics.brightness)
        shp = _normalize_sharpness(metrics.sharpness)
        image_quality = (bri + shp) / 2.0
        scoring_notes.append(
            "Learned IQA unavailable; image quality used brightness/sharpness fallback."
        )

    # Scene validity: use CLIP if available, default 75 (neutral)
    scene_validity = (
        _mean(metrics.scene_validity_scores)
        if metrics.scene_validity_scores
        else 75.0
    )
    if not metrics.scene_validity_scores:
        scoring_notes.append(
            "Scene validation unavailable; scene validity used neutral 75.0 fallback."
        )
    if not metrics.audio_details:
        scoring_notes.append(
            "Audio analysis unavailable or no audio extracted; audio quality scored as 0.0."
        )

    # Learned video quality (DOVER): fall back to per-frame image quality.
    if metrics.vqa_overall:
        video_quality = _mean(metrics.vqa_overall)
    else:
        video_quality = image_quality
        scoring_notes.append(
            "DOVER video-quality unavailable; video quality used image-quality fallback."
        )

    # Depth structure (Depth Anything V2): neutral 75 when unavailable.
    if metrics.depth_structure_scores:
        depth_structure = _mean(metrics.depth_structure_scores)
    else:
        depth_structure = 75.0
        scoring_notes.append(
            "Depth structure unavailable; depth structure used neutral 75.0 fallback."
        )

    metric_scores = {
        "image_quality": image_quality,
        "video_quality": video_quality,
        "stability": _normalize_stability(metrics.stability),
        "hand_detection_rate": _normalize_hand_detection_rate(
            metrics.hand_detection_rate
        ),
        "hand_landmark_quality": _normalize_hand_landmark_quality(
            metrics.hand_landmark_counts
        ),
        "tracking_continuity": metrics.tracking_continuity,
        "scene_validity": scene_validity,
        "anomaly_score": _mean(metrics.anomaly_scores) if metrics.anomaly_scores else 100.0,
        "blur_score": _mean(metrics.blur_scores) if metrics.blur_scores else 100.0,
        "temporal_consistency": metrics.temporal_consistency,
        "depth_structure": depth_structure,
        "audio_quality": metrics.audio_quality,
    }

    raw_metrics = {
        "brightness_mean": _mean(metrics.brightness),
        "sharpness_mean": _mean(metrics.sharpness),
        "stability_mean": _mean(metrics.stability),
        "hand_detection_rate": metrics.hand_detection_rate,
        "hand_confidence_mean": _mean(metrics.hand_confidence),
        "landmark_count_mean": _mean(metrics.hand_landmark_counts),
        "tracking_continuity": metrics.tracking_continuity,
        "total_frames": len(metrics.brightness),
        "iqa_mean": _mean(metrics.iqa_scores),
        "vqa_overall_mean": _mean(metrics.vqa_overall),
        "vqa_technical_mean": _mean(metrics.vqa_technical),
        "vqa_aesthetic_mean": _mean(metrics.vqa_aesthetic),
        "depth_structure_mean": _mean(metrics.depth_structure_scores),
        "anomaly_mean": _mean(metrics.anomaly_scores),
        "blur_mean": _mean(metrics.blur_scores),
        "scene_validity_mean": _mean(metrics.scene_validity_scores),
        "temporal_consistency": metrics.temporal_consistency,
        "temporal_flicker": metrics.temporal_flicker,
        "temporal_drops": len(metrics.temporal_quality_drops),
        "temporal_dupes": len(metrics.temporal_duplicates),
        "audio_quality": metrics.audio_quality,
        **{f"audio_{k}": v for k, v in metrics.audio_details.items()},
    }

    wd = w.as_dict()
    contributions = _weighted_contributions(metric_scores, wd)
    composite = max(0.0, min(100.0, sum(contributions.values())))

    return VideoScore(
        operator_id=video.operator_id,
        filename=video.filename,
        video_path=video.video_path,
        composite_score=round(composite, 1),
        grade=assign_grade(composite),
        metric_scores={k: round(v, 1) for k, v in metric_scores.items()},
        metric_weights={k: round(v, 4) for k, v in _normalized_weights(wd).items()},
        score_contributions={k: round(v, 2) for k, v in contributions.items()},
        raw_metrics={k: round(v, 2) for k, v in raw_metrics.items()},
        segment_scores=metrics.segment_scores,
        worst_issue=_identify_worst_issue(metric_scores),
        recommendations=_build_recommendations(metric_scores),
        scoring_notes=scoring_notes,
    )


def score_video(
    video: VideoFile,
    metrics: VideoMetrics,
    settings: BenchmarkSettings,
) -> VideoScore:
    """Compute composite score for a single video."""
    if settings.weights_version == "v2":
        return _score_v2(video, metrics, settings)
    return _score_v1(video, metrics, settings)
