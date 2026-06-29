"""Extract video segments using FFmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from math import ceil, floor
from pathlib import Path

from video_benchmark.acceleration import AccelerationInfo
from video_benchmark.config import SegmentSpec


def probe_video_duration(video_path: str, accel: AccelerationInfo) -> float | None:
    """Return video duration in seconds via ffprobe when available."""
    ffmpeg = Path(accel.ffmpeg_path).name if accel.ffmpeg_path else "ffmpeg"
    ffprobe = str(Path(accel.ffmpeg_path).with_name("ffprobe")) if accel.ffmpeg_path else "ffprobe"
    if ffmpeg != "ffmpeg" and not Path(ffprobe).exists():
        ffprobe = "ffprobe"

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    if duration <= 0:
        return None
    return duration


def effective_segments(
    requested_segments: list[SegmentSpec],
    duration_sec: float | None,
) -> list[SegmentSpec]:
    """Clip or replace requested windows so extraction works for any duration.

    Operator recordings are usually long, so the default windows target minutes
    2-4, 28-30, and 55-57. Shorter review clips should still be scorable; when
    all requested windows are beyond the clip, sample broad windows across the
    available duration instead of returning no data.
    """
    if duration_sec is None:
        return requested_segments

    clipped: list[SegmentSpec] = []
    for segment in requested_segments:
        start = max(0.0, min(float(segment.start_sec), duration_sec))
        end = max(start, min(float(segment.end_sec), duration_sec))
        if end - start >= 1.0:
            clipped.append(SegmentSpec(start_sec=floor(start), end_sec=ceil(end)))

    if clipped:
        return clipped

    if not requested_segments:
        return []

    if duration_sec <= 121.0:
        return [SegmentSpec(start_sec=0, end_sec=max(1, ceil(duration_sec)))]

    count = min(len(requested_segments), 3)
    window = min(120.0, max(5.0, duration_sec / max(count, 1)))
    max_start = max(0.0, duration_sec - window)

    fallback: list[SegmentSpec] = []
    seen: set[tuple[int, int]] = set()
    for idx in range(count):
        start = 0.0 if count == 1 else max_start * idx / (count - 1)
        end = min(duration_sec, start + window)
        spec = (floor(start), max(floor(start) + 1, ceil(end)))
        if spec not in seen:
            fallback.append(SegmentSpec(start_sec=spec[0], end_sec=spec[1]))
            seen.add(spec)
    return fallback


def extract_segment(
    video_path: str,
    segment: SegmentSpec,
    output_dir: Path,
    accel: AccelerationInfo,
    segment_index: int = 0,
) -> Path:
    """Extract a segment from a video file to a temporary mp4.

    Uses -ss before -i for fast seeking.
    """
    ffmpeg = accel.ffmpeg_path or "ffmpeg"
    duration = segment.end_sec - segment.start_sec
    output_file = output_dir / f"segment_{segment_index}.mp4"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(segment.start_sec),
        *accel.hwaccel_args,
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "copy",
        "-an",
        "-y",
        str(output_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg segment extraction failed for {video_path} "
            f"(segment {segment_index}): {result.stderr}"
        )
    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError(
            f"FFmpeg produced an empty segment for {video_path} "
            f"(segment {segment_index})"
        )
    return output_file


def extract_all_segments(
    video_path: str,
    segments: list[SegmentSpec],
    accel: AccelerationInfo,
    work_dir: Path | None = None,
) -> list[Path]:
    """Extract multiple segments from a video. Returns paths to segment files."""
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="vb_segments_"))

    results: list[Path] = []
    duration = probe_video_duration(video_path, accel)
    for i, seg in enumerate(effective_segments(segments, duration)):
        try:
            out = extract_segment(video_path, seg, work_dir, accel, i)
            results.append(out)
        except RuntimeError:
            # Segment might be past end of video — skip it
            continue
    return results
