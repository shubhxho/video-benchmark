"""UI-agnostic progress reporting for the pipeline.

`run_pipeline` drives a :class:`ProgressReporter` so the same processing loop can
back the plain Rich CLI, an interactive Textual TUI, or a silent batch run. The
default reporter reproduces the original ``rich.progress`` bar exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

if TYPE_CHECKING:
    from video_benchmark.pipeline.orchestrator import PipelineResult
    from video_benchmark.scoring.scorer import VideoScore
    from video_benchmark.sources.base import VideoFile


class ProgressReporter:
    """No-op progress reporter. Subclass to drive a UI from ``run_pipeline``."""

    def start(self, total: int) -> None:
        """Called once before processing, with the number of videos."""

    def video_started(self, video: VideoFile) -> None:
        """Called before a video begins processing (serial path only)."""

    def video_done(
        self,
        video: VideoFile,
        score: VideoScore | None,
        error: str | None,
    ) -> None:
        """Called when a video finishes (success carries ``score``, failure ``error``)."""

    def finish(self, result: PipelineResult) -> None:
        """Called once after all videos are processed."""


class RichProgressReporter(ProgressReporter):
    """Reproduces the original ``rich.progress`` bar shown during a CLI run."""

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        self._task_id: TaskID | None = None

    def start(self, total: int) -> None:
        self._progress.start()
        self._task_id = self._progress.add_task("Processing videos...", total=total)

    def video_started(self, video: VideoFile) -> None:
        if self._task_id is not None:
            self._progress.update(
                self._task_id, description=f"Processing {video.filename}"
            )

    def video_done(
        self,
        video: VideoFile,
        score: VideoScore | None,
        error: str | None,
    ) -> None:
        if self._task_id is not None:
            self._progress.advance(self._task_id)

    def finish(self, result: PipelineResult) -> None:
        self._progress.stop()
