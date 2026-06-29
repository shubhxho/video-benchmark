"""Resolve a video list from settings without CLI-specific error handling.

The Typer CLI resolver prints to a console and raises ``typer.Exit``; the TUI
needs a pure function it can call and surface errors inline in the form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from video_benchmark.sources.local import LocalVideoSource
from video_benchmark.sources.manifest import load_manifest
from video_benchmark.sources.s3 import S3VideoSource

if TYPE_CHECKING:
    from video_benchmark.config import BenchmarkSettings
    from video_benchmark.sources.base import VideoFile


def resolve_videos(settings: BenchmarkSettings) -> tuple[list[VideoFile], str | None]:
    """Return ``(videos, error)``. ``error`` is a human message when empty."""
    try:
        if settings.manifest:
            videos = load_manifest(settings.manifest, settings.path)
        elif settings.source == "s3":
            if not settings.bucket:
                return [], "A bucket name is required for the S3 source."
            videos = S3VideoSource(settings.bucket, settings.prefix).list_videos()
        else:
            if not settings.path:
                return [], "A directory path is required for the local source."
            videos = LocalVideoSource(settings.path).list_videos()
    except Exception as exc:  # noqa: BLE001 — surface any source error in the form
        return [], str(exc)

    if not videos:
        return [], "No videos found for the given source."
    return videos, None
