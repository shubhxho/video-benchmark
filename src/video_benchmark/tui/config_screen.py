"""huh-style interactive configuration form for the benchmark TUI.

Presents a keyboard-driven form (selects, text inputs, a toggle, a multi-select)
so a run can be configured entirely in the terminal instead of via CLI flags.
On submit it validates input, resolves the video list, and dismisses with a
:class:`RunConfig`; validation/resolution errors are shown inline so the form
stays open until the configuration is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Rule,
    Select,
    SelectionList,
    Static,
    Switch,
)

from video_benchmark.config import BenchmarkSettings
from video_benchmark.tui.resolve import resolve_videos

if TYPE_CHECKING:
    from video_benchmark.sources.base import VideoFile


@dataclass
class RunConfig:
    """The validated configuration produced by the form."""

    settings: BenchmarkSettings
    videos: list[VideoFile]


def _field(title: str, description: str) -> ComposeResult:
    """Yield a huh-style label + dim help line for the next widget."""
    yield Label(title, classes="field-label")
    yield Label(description, classes="field-desc")


class ConfigScreen(Screen[RunConfig | None]):
    """Interactive form collecting the settings for a benchmark run."""

    BINDINGS = [
        ("ctrl+r", "start", "Start run"),
        ("escape", "cancel", "Quit"),
    ]

    def __init__(self, initial: BenchmarkSettings) -> None:
        super().__init__()
        self._initial = initial

    def compose(self) -> ComposeResult:
        s = self._initial
        yield Header(show_clock=False)
        with VerticalScroll(id="form"):
            yield Static("Configure benchmark run", id="form-title")
            yield Static(
                "Fill in the fields, then press [b]Start[/b] (or ctrl+r). "
                "Tab/Shift+Tab move between fields.",
                id="form-help",
            )
            yield Rule()

            yield from _field("Source", "Where the videos come from.")
            yield Select(
                [("Local directory", "local"), ("S3 bucket", "s3")],
                value=s.source,
                allow_blank=False,
                id="source",
            )

            yield from _field("Local path", "Directory of videos (local source).")
            yield Input(
                value=str(s.path) if s.path else "",
                placeholder="./videos",
                id="path",
            )

            yield from _field(
                "S3 bucket / prefix", "Bucket name and optional key prefix (S3 source)."
            )
            with Horizontal(classes="row"):
                yield Input(value=s.bucket or "", placeholder="my-bucket", id="bucket")
                yield Input(value=s.prefix, placeholder="prefix/ (optional)", id="prefix")

            yield from _field("Manifest", "Optional CSV manifest; overrides source when set.")
            yield Input(
                value=str(s.manifest) if s.manifest else "",
                placeholder="manifest.csv (optional)",
                id="manifest",
            )

            yield Rule()

            yield from _field(
                "Scoring model", "v2 adds learned ML metrics; v1 is classical CV only."
            )
            yield Select(
                [("v2 — ML-enhanced", "v2"), ("v1 — classical", "v1")],
                value=s.weights_version,
                allow_blank=False,
                id="weights",
            )

            yield from _field("Workers", "Parallel videos (0 = auto-tune, GPU-aware).")
            yield Input(value=str(s.workers), type="integer", id="workers")

            yield from _field("Sample rate", "Frames per second to sample from each segment.")
            yield Input(value=str(s.sample_rate), type="integer", id="sample-rate")

            yield from _field("Segments", "Number of time windows to sample per video.")
            yield Input(value=str(s.segments), type="integer", id="segments")

            yield Rule()

            yield from _field("GPU acceleration", "Use VideoToolbox / MPS when available.")
            with Horizontal(classes="row switch-row"):
                yield Switch(value=not s.no_gpu, id="gpu")
                yield Label("enabled" if not s.no_gpu else "disabled", id="gpu-label")

            yield from _field("Extras", "Toggle with space; select multiple.")
            yield SelectionList[str](
                ("Generate HTML report", "report", s.report),
                ("Disable DOVER video-quality (faster)", "no_vqa", s.no_vqa),
                ("Disable depth-structure metric (faster)", "no_depth", s.no_depth),
                id="extras",
            )

            yield Static("", id="form-error")
            with Horizontal(id="form-buttons"):
                yield Button("Start ▶", variant="success", id="start")
                yield Button("Quit", variant="error", id="quit")
        yield Footer()

    # --- Dynamic affordances ----------------------------------------------

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "gpu":
            self.query_one("#gpu-label", Label).update(
                "enabled" if event.value else "disabled"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()
        elif event.button.id == "quit":
            self.action_cancel()

    # --- Actions -----------------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_start(self) -> None:
        settings, err = self._build_settings()
        if err is not None:
            self._error(err)
            return
        assert settings is not None
        videos, verr = resolve_videos(settings)
        if verr is not None:
            self._error(verr)
            return
        self.dismiss(RunConfig(settings=settings, videos=videos))

    # --- Helpers -----------------------------------------------------------

    def _error(self, message: str) -> None:
        self.query_one("#form-error", Static).update(f"[b red]✗ {message}[/b red]")

    def _value(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def _int(self, widget_id: str, default: int) -> int:
        raw = self._value(widget_id)
        try:
            return int(raw)
        except ValueError:
            return default

    def _build_settings(self) -> tuple[BenchmarkSettings | None, str | None]:
        source = cast(Literal["local", "s3"], self.query_one("#source", Select).value)
        weights = cast(Literal["v1", "v2"], self.query_one("#weights", Select).value)
        path_str = self._value("path")
        manifest_str = self._value("manifest")
        bucket = self._value("bucket") or None
        prefix = self._value("prefix")
        extras = set(self.query_one("#extras", SelectionList).selected)
        gpu_on = self.query_one("#gpu", Switch).value

        path = Path(path_str) if path_str else None
        manifest = Path(manifest_str) if manifest_str else None

        if manifest is not None and not manifest.exists():
            return None, f"Manifest not found: {manifest}"
        if manifest is None:
            if source == "local":
                if path is None:
                    return None, "Enter a local directory path."
                if not path.exists() or not path.is_dir():
                    return None, f"Path is not a directory: {path}"
            elif source == "s3" and not bucket:
                return None, "Enter an S3 bucket name."

        settings = BenchmarkSettings(
            source=source,
            path=path,
            bucket=bucket,
            prefix=prefix,
            manifest=manifest,
            workers=self._int("workers", 0),
            sample_rate=max(1, self._int("sample-rate", 1)),
            segments=max(1, self._int("segments", 3)),
            no_gpu=not gpu_on,
            weights_version=weights,
            report="report" in extras,
            no_vqa="no_vqa" in extras,
            no_depth="no_depth" in extras,
        )
        return settings, None
