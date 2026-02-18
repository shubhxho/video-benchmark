"""Main pipeline orchestration — process videos end-to-end."""

from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from video_benchmark.acceleration import AccelerationInfo, detect_acceleration
from video_benchmark.config import BenchmarkSettings, detect_available_models
from video_benchmark.metrics.anomalies import AnomalyDetector
from video_benchmark.metrics.blur import BlurClassifier
from video_benchmark.metrics.brightness import BrightnessMetric
from video_benchmark.metrics.continuity import ContinuityMetric
from video_benchmark.metrics.hand_detection import HandDetectionMetric
from video_benchmark.metrics.sharpness import SharpnessMetric
from video_benchmark.metrics.stability import StabilityMetric
from video_benchmark.metrics.temporal import TemporalConsistencyMetric
from video_benchmark.pipeline.frame_sampler import extract_frames_cv2
from video_benchmark.pipeline.segment_extractor import extract_all_segments
from video_benchmark.scoring.aggregator import aggregate_operators
from video_benchmark.scoring.scorer import VideoScore, score_video
from video_benchmark.sources.base import VideoFile

logger = logging.getLogger(__name__)


def _mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


@dataclass
class VideoMetrics:
    # --- V1 metrics (always available) ---
    brightness: list[float] = field(default_factory=list)
    sharpness: list[float] = field(default_factory=list)
    stability: list[float] = field(default_factory=list)
    hand_detection_rate: float = 0.0
    hand_confidence: list[float] = field(default_factory=list)
    hand_landmark_counts: list[int] = field(default_factory=list)
    tracking_continuity: float = 0.0
    detection_flags: list[bool] = field(default_factory=list)
    segment_scores: list[dict] = field(default_factory=list)

    # --- V2 metrics (new) ---
    iqa_scores: list[float] = field(default_factory=list)
    anomaly_scores: list[float] = field(default_factory=list)
    anomaly_flags: list[list[str]] = field(default_factory=list)
    blur_scores: list[float] = field(default_factory=list)
    blur_types: list[str] = field(default_factory=list)
    temporal_consistency: float = 100.0
    temporal_quality_drops: list[int] = field(default_factory=list)
    temporal_flicker: float = 100.0
    temporal_duplicates: list[int] = field(default_factory=list)
    scene_validity_scores: list[float] = field(default_factory=list)
    audio_quality: float = 0.0
    audio_details: dict = field(default_factory=dict)

    # Frame cache for HTML report
    best_frame: np.ndarray | None = None
    worst_frame: np.ndarray | None = None
    best_frame_score: float = -1.0
    worst_frame_score: float = 999.0


def process_single_video(
    video: VideoFile,
    settings: BenchmarkSettings,
    accel: AccelerationInfo,
) -> tuple[VideoFile, VideoMetrics | None, str | None]:
    """Process a single video through the full metric pipeline.

    Returns (video, metrics, error_message).
    """
    try:
        segments = settings.segment_specs()
        use_v2 = settings.weights_version == "v2"
        available = detect_available_models() if use_v2 else {}

        with tempfile.TemporaryDirectory(prefix="vb_") as tmpdir:
            work_dir = Path(tmpdir)
            segment_paths = extract_all_segments(
                video.video_path, segments, accel, work_dir
            )

            if not segment_paths:
                return video, None, "No segments could be extracted"

            metrics = VideoMetrics()
            all_detection_flags: list[bool] = []

            # --- Optional ML models (lazy init, created once) ---
            iqa_metric = None
            if use_v2 and available.get("pyiqa"):
                from video_benchmark.metrics.iqa import LearnedIQAMetric
                iqa_metric = LearnedIQAMetric("topiq_nr")

            raft_metric = None
            if use_v2 and available.get("torchvision_raft"):
                from video_benchmark.metrics.deep_flow import RAFTStabilityMetric
                raft_metric = RAFTStabilityMetric()

            yolo_metric = None
            if use_v2 and available.get("ultralytics"):
                from video_benchmark.metrics.yolo_hands import YOLOHandMetric
                yolo_metric = YOLOHandMetric()

            scene_validator = None
            if use_v2 and available.get("open_clip"):
                from video_benchmark.metrics.scene import SceneValidator
                scene_validator = SceneValidator()

            anomaly_det = AnomalyDetector()
            blur_clf = BlurClassifier()
            temporal_metric = TemporalConsistencyMetric()

            for seg_idx, seg_path in enumerate(segment_paths):
                frames = extract_frames_cv2(seg_path, settings.sample_rate)
                if not frames:
                    continue

                # === V1 metrics (always run) ===
                bm = BrightnessMetric()
                brightness_scores = [bm.compute(f) for f in frames]
                metrics.brightness.extend(brightness_scores)

                sm = SharpnessMetric()
                sharpness_scores = [sm.compute(f) for f in frames]
                metrics.sharpness.extend(sharpness_scores)

                # Stability: RAFT if available, else Farneback
                stab = StabilityMetric()
                for i in range(1, len(frames)):
                    if raft_metric is not None:
                        val = raft_metric.compute_flow(frames[i - 1], frames[i])
                        if val is not None:
                            metrics.stability.append(val)
                            continue
                    metrics.stability.append(
                        stab.compute_flow(frames[i - 1], frames[i])
                    )

                # Hand detection: YOLO if available, else MediaPipe
                seg_detections: list[bool] = []
                if yolo_metric is not None:
                    for frame in frames:
                        result = yolo_metric.detect(frame)
                        if result is not None:
                            seg_detections.append(result.detected)
                            all_detection_flags.append(result.detected)
                            if result.detected:
                                metrics.hand_confidence.append(result.confidence)
                                metrics.hand_landmark_counts.append(result.landmark_count)
                        else:
                            hd = HandDetectionMetric()
                            mp_r = hd.detect(frame)
                            seg_detections.append(mp_r.detected)
                            all_detection_flags.append(mp_r.detected)
                            if mp_r.detected:
                                metrics.hand_confidence.append(mp_r.confidence)
                                metrics.hand_landmark_counts.append(mp_r.landmark_count)
                            hd.close()
                else:
                    hd = HandDetectionMetric()
                    for frame in frames:
                        result = hd.detect(frame)
                        seg_detections.append(result.detected)
                        all_detection_flags.append(result.detected)
                        if result.detected:
                            metrics.hand_confidence.append(result.confidence)
                            metrics.hand_landmark_counts.append(result.landmark_count)
                    hd.close()

                # === V2 metrics ===

                # Learned IQA
                if iqa_metric is not None:
                    for frame in frames:
                        val = iqa_metric.compute(frame)
                        if val is not None:
                            metrics.iqa_scores.append(val)

                # Anomaly detection (pure OpenCV — always run)
                for frame in frames:
                    metrics.anomaly_scores.append(
                        anomaly_det.compute_anomaly_score(frame)
                    )
                    metrics.anomaly_flags.append(
                        anomaly_det.detect_anomalies(frame)
                    )

                # Blur classification (pure OpenCV — always run)
                for frame in frames:
                    br = blur_clf.classify(frame)
                    metrics.blur_scores.append(max(0.0, 100.0 - br.severity))
                    metrics.blur_types.append(br.blur_type)

                # Scene validation (sample ~10% — expensive)
                if scene_validator is not None:
                    step = max(1, len(frames) // 10) if len(frames) > 10 else 1
                    for i in range(0, len(frames), step):
                        sv = scene_validator.compute_validity_score(frames[i])
                        if sv is not None:
                            metrics.scene_validity_scores.append(sv)

                # Temporal consistency per segment
                temp = temporal_metric.compute(frames)
                metrics.temporal_quality_drops.extend(temp.quality_drops)
                metrics.temporal_duplicates.extend(temp.duplicate_frames)

                # Track best/worst frames for report
                for idx, frame in enumerate(frames):
                    q = sharpness_scores[idx] if idx < len(sharpness_scores) else 0
                    if q > metrics.best_frame_score:
                        metrics.best_frame_score = q
                        metrics.best_frame = frame.copy()
                    if q < metrics.worst_frame_score:
                        metrics.worst_frame_score = q
                        metrics.worst_frame = frame.copy()

                seg_score = {
                    "segment": seg_idx,
                    "frames": len(frames),
                    "brightness_mean": _mean(brightness_scores),
                    "sharpness_mean": _mean(sharpness_scores),
                    "hand_rate": (
                        sum(seg_detections) / len(seg_detections) if seg_detections else 0
                    ),
                    "anomaly_mean": _mean(metrics.anomaly_scores[-len(frames):]),
                    "temporal_consistency": temp.consistency_score,
                    "flicker_score": temp.flicker_score,
                }
                metrics.segment_scores.append(seg_score)

            # Aggregate temporal scores
            seg_temps = [s.get("temporal_consistency", 100.0) for s in metrics.segment_scores]
            metrics.temporal_consistency = _mean(seg_temps) if seg_temps else 100.0
            seg_flick = [s.get("flicker_score", 100.0) for s in metrics.segment_scores]
            metrics.temporal_flicker = _mean(seg_flick) if seg_flick else 100.0

            metrics.detection_flags = all_detection_flags
            if all_detection_flags:
                metrics.hand_detection_rate = sum(all_detection_flags) / len(all_detection_flags)

            cont = ContinuityMetric()
            metrics.tracking_continuity = cont.compute_from_flags(all_detection_flags)

            # Audio quality (once per video)
            if use_v2:
                from video_benchmark.metrics.audio import AudioQualityMetric
                audio_m = AudioQualityMetric()
                audio_r = audio_m.analyze(
                    video.video_path, work_dir, accel.ffmpeg_path or "ffmpeg"
                )
                if audio_r is not None:
                    metrics.audio_quality = audio_r.overall_score
                    metrics.audio_details = {
                        "loudness_lufs": audio_r.loudness_lufs,
                        "snr_db": audio_r.snr_db,
                        "silence_pct": audio_r.silence_pct,
                        "has_wind_noise": audio_r.has_wind_noise,
                    }

        return video, metrics, None

    except Exception as e:
        logger.exception(f"Error processing {video.filename}")
        return video, None, str(e)


def _resolve_worker_count(
    requested_workers: int,
    total_videos: int,
    accel: AccelerationInfo,
) -> int:
    """Resolve effective worker count.

    Auto mode (`requested_workers <= 0`) scales concurrency aggressively on
    Apple GPU paths to keep decode and metric stages saturated.
    """
    if total_videos <= 1:
        return 1

    if requested_workers > 0:
        return max(1, min(requested_workers, total_videos))

    cpu = os.cpu_count() or 4
    if accel.videotoolbox or accel.mps_available:
        auto_workers = min(total_videos, max(2, cpu * 2))
    else:
        auto_workers = min(total_videos, max(1, cpu))
    return max(1, auto_workers)


@dataclass
class PipelineResult:
    scores: list[VideoScore]
    failed: list[tuple[VideoFile, str]]
    operator_rankings: list[dict]
    frame_cache: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    run_info: dict[str, str | int | bool] = field(default_factory=dict)


def run_pipeline(
    videos: list[VideoFile],
    settings: BenchmarkSettings,
) -> PipelineResult:
    """Run the full benchmark pipeline on all videos."""
    accel = detect_acceleration(force_no_gpu=settings.no_gpu)
    worker_count = _resolve_worker_count(settings.workers, len(videos), accel)
    executor_mode = "threads" if worker_count > 1 else "serial"

    # Log available models
    if settings.weights_version == "v2":
        models = detect_available_models()
        for name, avail in models.items():
            status = "enabled" if avail else "not installed"
            logger.info(f"Model {name}: {status}")

    scores: list[VideoScore] = []
    failed: list[tuple[VideoFile, str]] = []
    frame_cache: dict[str, dict[str, np.ndarray]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing videos...", total=len(videos))

        if worker_count <= 1:
            for video in videos:
                progress.update(task, description=f"Processing {video.filename}")
                v, metrics, error = process_single_video(video, settings, accel)
                if metrics and error is None:
                    vs = score_video(v, metrics, settings)
                    scores.append(vs)
                    fc: dict[str, np.ndarray] = {}
                    if metrics.best_frame is not None:
                        fc["best"] = metrics.best_frame
                    if metrics.worst_frame is not None:
                        fc["worst"] = metrics.worst_frame
                    if fc:
                        frame_cache[v.filename] = fc
                else:
                    failed.append((v, error or "Unknown error"))
                progress.advance(task)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(process_single_video, video, settings, accel): video
                    for video in videos
                }
                for future in as_completed(futures):
                    v, metrics, error = future.result()
                    if metrics and error is None:
                        vs = score_video(v, metrics, settings)
                        scores.append(vs)
                        fc = {}
                        if metrics.best_frame is not None:
                            fc["best"] = metrics.best_frame
                        if metrics.worst_frame is not None:
                            fc["worst"] = metrics.worst_frame
                        if fc:
                            frame_cache[v.filename] = fc
                    else:
                        failed.append((v, error or "Unknown error"))
                    progress.advance(task)

    operator_rankings = aggregate_operators(scores)

    return PipelineResult(
        scores=scores,
        failed=failed,
        operator_rankings=operator_rankings,
        frame_cache=frame_cache,
        run_info={
            "executor_mode": executor_mode,
            "workers_requested": settings.workers,
            "workers_used": worker_count,
            "videotoolbox": accel.videotoolbox,
            "mps_available": accel.mps_available,
            "weights_version": settings.weights_version,
        },
    )
