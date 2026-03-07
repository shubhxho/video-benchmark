"""Video compression utilities with optional LLM guidance."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from video_benchmark.acceleration import AccelerationInfo, detect_acceleration, require_ffmpeg


@dataclass
class VideoProbe:
    width: int
    height: int
    fps: float
    bitrate_kbps: float | None
    duration_s: float | None
    size_bytes: int


@dataclass
class CompressionPlan:
    codec: str
    preset: str
    crf: int | None
    video_bitrate: str | None
    scale: str | None
    audio_bitrate: str = "96k"


@dataclass
class CompressionResult:
    source: Path
    output: Path
    source_size: int
    output_size: int

    @property
    def ratio(self) -> float:
        return self.output_size / self.source_size if self.source_size else 0.0


def probe_video(path: Path, ffprobe_path: str | None = None) -> VideoProbe:
    ffprobe = ffprobe_path or "ffprobe"
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,bit_rate:format=size,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")

    info = json.loads(result.stdout)
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format", {})

    fps_str = stream.get("avg_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0

    bitrate = stream.get("bit_rate") or fmt.get("bit_rate")
    bitrate_kbps = float(bitrate) / 1000.0 if bitrate else None

    duration_val = fmt.get("duration")
    try:
        duration_s = float(duration_val) if duration_val is not None else None
    except (TypeError, ValueError):
        duration_s = None

    size_bytes = int(fmt.get("size", 0))

    return VideoProbe(
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=fps,
        bitrate_kbps=bitrate_kbps,
        duration_s=duration_s,
        size_bytes=size_bytes,
    )


def _default_plan(probe: VideoProbe, codec: str, crf: int | None, scale: str | None) -> CompressionPlan:
    target_codec = codec.lower()
    if target_codec in {"h265", "hevc"}:
        codec_name = "libx265"
        default_crf = 27
    elif target_codec in {"av1", "svt-av1"}:
        codec_name = "libsvtav1"
        default_crf = 32
    else:
        codec_name = "libx264"
        default_crf = 23

    chosen_crf = crf if crf is not None else default_crf

    target_bitrate = None
    if probe.bitrate_kbps:
        # Aim for ~55% of the current bitrate by default.
        target_bitrate = f"{int(max(probe.bitrate_kbps * 0.55, 500))}k"

    auto_scale = scale
    if scale is None and max(probe.width, probe.height) > 1920:
        # Downscale very high-res footage to 1080p while keeping aspect ratio.
        auto_scale = "-2:1080" if probe.height >= probe.width else "1920:-2"

    return CompressionPlan(
        codec=codec_name,
        preset="medium",
        crf=chosen_crf,
        video_bitrate=target_bitrate,
        scale=auto_scale,
    )


def _encoder_for_plan(plan: CompressionPlan, accel: AccelerationInfo) -> tuple[str, list[str]]:
    # Prefer Apple VideoToolbox when available and the target codec supports it.
    if accel.videotoolbox:
        if plan.codec == "libx265":
            return "hevc_videotoolbox", ["-tag:v", "hvc1"]
        if plan.codec == "libx264":
            return "h264_videotoolbox", []

    return plan.codec, ["-preset", plan.preset]


def compress_video(
    video_path: Path,
    output_dir: Path,
    plan: CompressionPlan,
    accel: AccelerationInfo | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    ffmpeg_path: str | None = None,
) -> CompressionResult:
    ffmpeg = ffmpeg_path or require_ffmpeg()
    accel = accel or detect_acceleration()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = video_path.suffix or ".mp4"
    out_path = output_dir / f"{video_path.stem}_compressed{suffix}"

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out_path}")

    filters: List[str] = []
    if plan.scale:
        filters.append(f"scale={plan.scale}")

    encoder, encoder_extras = _encoder_for_plan(plan, accel)

    cmd: list[str] = [ffmpeg, "-hide_banner"]
    if not verbose:
        cmd += ["-loglevel", "error"]
    cmd += accel.hwaccel_args
    cmd += ["-i", str(video_path)]
    if filters:
        cmd += ["-vf", ",".join(filters)]

    cmd += ["-c:v", encoder]
    if plan.crf is not None and not encoder.endswith("videotoolbox"):
        cmd += ["-crf", str(plan.crf)]
    if plan.video_bitrate:
        cmd += ["-b:v", plan.video_bitrate, "-maxrate", plan.video_bitrate, "-bufsize", "2M"]
    cmd += encoder_extras

    cmd += ["-c:a", "aac", "-b:a", plan.audio_bitrate]
    cmd += ["-movflags", "+faststart"]
    cmd += ["-map_metadata", "0", "-map_chapters", "-1"]
    cmd += ["-y" if overwrite else "-n", str(out_path)]

    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr}")

    source_size = video_path.stat().st_size
    output_size = out_path.stat().st_size
    return CompressionResult(video_path, out_path, source_size, output_size)


def select_plan(
    probe: VideoProbe,
    codec: str = "h265",
    crf: int | None = None,
    scale: str | None = None,
    use_llm: bool = False,
    api_key: str | None = None,
) -> CompressionPlan:
    base_plan = _default_plan(probe, codec, crf, scale)
    if not use_llm:
        return base_plan

    try:
        from video_benchmark.llm.compression_advisor import advise_compression
    except ImportError:
        return base_plan

    advice = advise_compression(probe, base_plan, api_key=api_key)
    return advice or base_plan


def find_videos(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".mp4":
        yield root
        return
    for mp4 in sorted(root.rglob("*.mp4")):
        yield mp4


def human_size(bytes_val: int) -> str:
    if bytes_val <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = min(int(math.log(bytes_val, 1024)), len(units) - 1)
    value = bytes_val / (1024 ** idx)
    return f"{value:.1f}{units[idx]}"

