"""Textual TUI application for scoring operator videos with a live dashboard."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import TYPE_CHECKING

from rich.table import Table
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    ProgressBar,
    Static,
    TabbedContent,
    TabPane,
)

from video_benchmark.benchmark.runner import BenchmarkReport, synthetic_benchmark
from video_benchmark.pipeline.orchestrator import PipelineResult, run_pipeline
from video_benchmark.pipeline.reporting import ProgressReporter
from video_benchmark.tui.styles import APP_CSS

if TYPE_CHECKING:
    from video_benchmark.config import BenchmarkSettings
    from video_benchmark.scoring.scorer import VideoScore
    from video_benchmark.sources.base import VideoFile

GRADE_STYLES = {
    "A": "bold green",
    "B": "bold cyan",
    "C": "bold yellow",
    "D": "bold red",
    "F": "bold bright_red",
}


def _grade_markup(grade: str) -> str:
    style = GRADE_STYLES.get(grade, "")
    return f"[{style}]{grade}[/{style}]" if style else grade


class TextualReporter(ProgressReporter):
    """Bridges run_pipeline progress (worker thread) to the Textual app."""

    def __init__(self, app: BenchmarkApp) -> None:
        self._app = app

    def start(self, total: int) -> None:
        self._app.call_from_thread(self._app.report_start, total)

    def video_done(
        self,
        video: VideoFile,
        score: VideoScore | None,
        error: str | None,
    ) -> None:
        self._app.call_from_thread(self._app.report_video_done, video, score, error)

    def finish(self, result: PipelineResult) -> None:
        self._app.call_from_thread(self._app.report_finish, result)


class BenchmarkApp(App[None]):
    """Interactive dashboard that runs the benchmark pipeline live."""

    CSS = APP_CSS
    TITLE = "Video Benchmark"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("b", "run_bench", "Run perf benchmark"),
        ("d", "toggle_dark", "Toggle theme"),
    ]

    def __init__(
        self,
        videos: list[VideoFile],
        settings: BenchmarkSettings,
    ) -> None:
        super().__init__()
        self._videos = videos
        self._settings = settings
        self._scores_by_name: dict[str, VideoScore] = {}
        self._result: PipelineResult | None = None
        self._done = 0

    # --- Layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        accel = "GPU off" if self._settings.no_gpu else "GPU auto"
        yield Static(
            f"[b]{len(self._videos)}[/b] videos  •  weights="
            f"[b]{self._settings.weights_version}[/b]  •  workers="
            f"[b]{self._settings.workers or 'auto'}[/b]  •  {accel}",
            id="runinfo",
        )
        yield ProgressBar(total=len(self._videos), id="pb")
        with TabbedContent(initial="live"):
            with TabPane("Live", id="live"):
                yield DataTable(id="live-table", cursor_type="row")
                yield Static("Select a video row for details.", id="detail")
            with TabPane("Operators", id="operators"):
                yield VerticalScroll(Static(id="op-table"))
            with TabPane("Metrics", id="metrics"):
                yield VerticalScroll(Static(id="metric-table"))
            with TabPane("Issues", id="issues"):
                yield VerticalScroll(Static(id="issue-table"))
            with TabPane("Failed", id="failed"):
                yield VerticalScroll(Static(id="failed-table"))
            with TabPane("Benchmark", id="benchmark"):
                yield VerticalScroll(
                    Static("Press 'b' to run the perf benchmark.", id="bench-table")
                )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#live-table", DataTable)
        table.add_columns("Video", "Operator", "Score", "Grade", "Worst issue")
        self._run_pipeline_worker()

    # --- Worker ------------------------------------------------------------

    @work(thread=True, exclusive=True)
    def _run_pipeline_worker(self) -> None:
        run_pipeline(self._videos, self._settings, reporter=TextualReporter(self))

    @work(thread=True, group="bench")
    def _run_bench_worker(self) -> None:
        report = synthetic_benchmark()
        self.call_from_thread(self._show_bench, report)

    # --- Reporter callbacks (run on the main thread) -----------------------

    def report_start(self, total: int) -> None:
        self.query_one("#pb", ProgressBar).update(total=total)

    def report_video_done(
        self,
        video: VideoFile,
        score: VideoScore | None,
        error: str | None,
    ) -> None:
        self._done += 1
        self.query_one("#pb", ProgressBar).advance(1)
        table = self.query_one("#live-table", DataTable)
        if score is not None:
            self._scores_by_name[score.filename] = score
            table.add_row(
                score.filename[:30],
                score.operator_id[:14],
                f"{score.composite_score:.1f}",
                _grade_markup(score.grade),
                score.worst_issue if score.worst_issue != "none" else "—",
                key=score.filename,
            )
        else:
            table.add_row(
                video.filename[:30],
                getattr(video, "operator_id", "—")[:14],
                "[red]FAIL[/red]",
                "[red]F[/red]",
                (error or "error")[:30],
            )

    def report_finish(self, result: PipelineResult) -> None:
        self._result = result
        self.sub_title = (
            f"{len(result.scores)} scored / {len(result.failed)} failed"
        )
        self._populate_operators(result)
        self._populate_metrics(result)
        self._populate_issues(result)
        self._populate_failed(result)

    # --- Summary tab population --------------------------------------------

    def _populate_operators(self, result: PipelineResult) -> None:
        table = Table(title="Operator Ranking", expand=True)
        table.add_column("Rank", justify="right")
        table.add_column("Operator")
        table.add_column("Score", justify="right")
        table.add_column("Grade", justify="center")
        table.add_column("Videos", justify="right")
        table.add_column("Worst issue")
        for r in result.operator_rankings:
            table.add_row(
                str(r["rank"]),
                r["operator_id"],
                f"{r['final_score']:.1f}",
                _grade_markup(r["grade"]),
                str(r["video_count"]),
                r["worst_issue"] if r["worst_issue"] != "none" else "—",
            )
        self.query_one("#op-table", Static).update(table)

    def _populate_metrics(self, result: PipelineResult) -> None:
        scores = result.scores
        metric_names = sorted({k for s in scores for k in s.metric_scores})
        table = Table(title="Metric Overview", expand=True)
        table.add_column("Metric")
        table.add_column("Avg", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        for metric in metric_names:
            vals = [s.metric_scores[metric] for s in scores if metric in s.metric_scores]
            if not vals:
                continue
            avg = mean(vals)
            style = "green" if avg >= 70 else "yellow" if avg >= 40 else "red"
            table.add_row(
                metric,
                f"[{style}]{avg:.1f}[/{style}]",
                f"{min(vals):.1f}",
                f"{max(vals):.1f}",
            )
        self.query_one("#metric-table", Static).update(table)

    def _populate_issues(self, result: PipelineResult) -> None:
        issues = [s.worst_issue for s in result.scores if s.worst_issue != "none"]
        table = Table(title="Common Issues", expand=True)
        table.add_column("Issue")
        table.add_column("Videos", justify="right")
        table.add_column("Pct", justify="right")
        total = len(result.scores) or 1
        for issue, count in Counter(issues).most_common(12):
            table.add_row(issue, str(count), f"{count / total * 100:.0f}%")
        if not issues:
            table.add_row("none", "0", "0%")
        self.query_one("#issue-table", Static).update(table)

    def _populate_failed(self, result: PipelineResult) -> None:
        table = Table(title=f"Failed Videos ({len(result.failed)})", expand=True)
        table.add_column("Video")
        table.add_column("Operator")
        table.add_column("Error")
        for video, error in result.failed:
            table.add_row(
                video.filename,
                getattr(video, "operator_id", "—"),
                f"[red]{str(error)[:70]}[/red]",
            )
        if not result.failed:
            table.add_row("—", "—", "No failures")
        self.query_one("#failed-table", Static).update(table)

    # --- Drill-down --------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        score = self._scores_by_name.get(event.row_key.value)
        if score is None:
            return
        title = f"{score.filename} — {score.composite_score:.1f} ({score.grade})"
        table = Table(title=title, expand=True)
        table.add_column("Metric")
        table.add_column("Score", justify="right")
        table.add_column("Weight", justify="right")
        for metric, value in sorted(score.metric_scores.items(), key=lambda kv: kv[1]):
            weight = score.metric_weights.get(metric, 0.0) * 100
            style = "green" if value >= 70 else "yellow" if value >= 40 else "red"
            table.add_row(metric, f"[{style}]{value:.1f}[/{style}]", f"{weight:.1f}%")
        self.query_one("#detail", Static).update(table)

    # --- Perf benchmark tab -------------------------------------------------

    def action_run_bench(self) -> None:
        self.query_one("#bench-table", Static).update("Running synthetic benchmark…")
        self._run_bench_worker()

    def _show_bench(self, report: BenchmarkReport) -> None:
        table = Table(title=f"Per-stage timing ({report.frames} synthetic frames)", expand=True)
        table.add_column("Stage")
        table.add_column("ms/frame", justify="right")
        table.add_column("frames/sec", justify="right")
        for s in report.stages:
            table.add_row(s.stage, f"{s.ms_per_frame:.3f}", f"{s.fps:.1f}")
        self.query_one("#bench-table", Static).update(table)

    def action_toggle_dark(self) -> None:
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"
