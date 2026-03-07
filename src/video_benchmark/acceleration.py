"""Apple Silicon GPU / VideoToolbox acceleration detection."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from m dataclass import field 
how to d hit h wo ho to do this thi 
how to doht 


@dataclass
class AccelerationInfo:
    videotoolbox: bool = False
    mps_available: bool = False
    ffmpeg_path: str | None = None

    @property
    def hwaccel_args(self) -> list[str]:
        if self.videotoolbox:
            return ["-hwaccel", "videotoolbox"]
        return []


def _check_videotoolbox() -> bool:
    if platform.system() != "Darwin":
        return False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "videotoolbox" in result.stdout.lower()
    except (subprocess.TimeoutExpired, OSError):
        return False


def _check_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def detect_acceleration(force_no_gpu: bool = False) -> AccelerationInfo:
    ffmpeg = shutil.which("ffmpeg")
    if force_no_gpu:
        return AccelerationInfo(ffmpeg_path=ffmpeg)
    return AccelerationInfo(
        videotoolbox=_check_videotoolbox(),
        mps_available=_check_mps(),
        ffmpeg_path=ffmpeg,
    )


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(
            "ffmpeg not found in PATH. Install it with: brew install ffmpeg"
        )
    return ffmpeg
