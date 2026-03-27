"""Rich console output — summary tables and grade distribution."""

from __future__ import annotations

from collections import Counter
from statistics import mean

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from video_benchmark.scoring.grader import grade_description
from video_benchmark.scoring.scorer import VideoScore

console = Console()

GRADE_STYLES = {
    "A": "bold green",
    "B": "bold blue",
    "C": "bold yellow",
    "D": "bold red",
    "F": "bold bright_red",
}

GRADE_BAR_CHARS = {
    "A": "[green]█[/green]",
    "B": "[blue]█[/blue]",
    "C": "[yellow]█[/yellow]",
    "D": "[red]█[/red]",
    "F": "[bright_red]█[/bright_red]",
}


def _styled_grade(grade: str) -> str:
    style = GRADE_STYLES.get(grade, "")
    return f"[{style}]{grade}[/{style}]"


def print_summary(
    scores: list[VideoScore],
    rankings: list[dict],
    failed: list[tuple],
    elapsed_seconds: float,
    run_info: dict[str, str | int | bool] | None = None,
) -> None:
    """Print a rich summary of benchmark results."""
    console.print()

    # Run info panel
    total = len(scores) + len(failed)
    info_items = [
        f"[bold]{len(scores)}[/bold] of {total} videos scored",
    ]
    if failed:
        info_items.append(f"[red]{len(failed)}[/red] failed")
    info_items.append(f"[bold]{len(rankings)}[/bold] operators")
    info_items.append(f"[bold]{elapsed_seconds:.1f}s[/bold] elapsed")

    if run_info:
        mode = run_info.get("executor_mode", "unknown")
        workers = run_info.get("workers_used", "?")
        vt = "✓" if run_info.get("videotoolbox") else "✗"
        mps = "✓" if run_info.get("mps_available") else "✗"
        wv = run_info.get("weights_version", "?")
        info_items.append(
            f"mode=[bold]{mode}[/bold]  workers=[bold]{workers}[/bold]  "
            f"VideoToolbox={vt}  MPS={mps}  weights=[bold]{wv}[/bold]"
        )

    console.print(
        Panel(
            "\n".join(info_items),
            title="[bold]Video Benchmark Results[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()

    # Grade distribution
    _print_grade_distribution(scores)
    _print_metric_overview(scores)

    # Top / Bottom operators
    if rankings:
        _print_operator_table("Top Operators", rankings[:10])
        if len(rankings) > 10:
            _print_operator_table("Needs Attention", rankings[-10:])

    _print_video_insights(scores)
    _print_operator_metric_breakdown(scores, rankings)
    _print_common_issues(scores)

    if failed:
        _print_failed(failed)


def _print_grade_distribution(scores: list[VideoScore]) -> None:
    grade_counts = Counter(s.grade for s in scores)
    total = len(scores) or 1

    table = Table(
        title="Grade Distribution",
        show_lines=False,
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Grade", width=6, justify="center")
    table.add_column("Bar", width=32)
    table.add_column("Count", width=6, justify="right")
    table.add_column("Pct", width=6, justify="right")
    table.add_column("Description", width=26)

    for grade in ["A", "B", "C", "D", "F"]:
        count = grade_counts.get(grade, 0)
        pct = count / total * 100
        bar_len = max(0, int(pct / 3.2))
        bar_char = GRADE_BAR_CHARS[grade]
        bar = bar_char * bar_len
        desc = grade_description(grade)
        table.add_row(
            _styled_grade(grade),
            bar,
            str(count),
            f"{pct:.0f}%",
            f"[dim]{desc}[/dim]",
        )

    console.print(table)
    console.print()


def _print_operator_table(title: str, rankings: list[dict]) -> None:
    table = Table(title=title, show_lines=False, row_styles=["", "dim"])
    table.add_column("Rank", style="dim", width=5, justify="right")
    table.add_column("Operator", width=16)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Grade", justify="center", width=6)
    table.add_column("Videos", justify="right", width=7)
    table.add_column("Usable", justify="right", width=7)
    table.add_column("Issue", width=22)

    for r in rankings:
        table.add_row(
            str(r["rank"]),
            r["operator_id"],
            f"{r['final_score']:.1f}",
            _styled_grade(r["grade"]),
            str(r["video_count"]),
            r["usable_pct"],
            r["worst_issue"] if r["worst_issue"] != "none" else "[dim]—[/dim]",
        )
    console.print(table)
    console.print()


def _print_common_issues(scores: list[VideoScore]) -> None:
    issues = [s.worst_issue for s in scores if s.worst_issue != "none"]
    if not issues:
        return

    counter = Counter(issues)
    table = Table(title="Common Issues", show_lines=False)
    table.add_column("Issue", width=28)
    table.add_column("Videos", justify="right", width=8)
    table.add_column("Pct", justify="right", width=8)

    total = len(scores) or 1
    for issue, count in counter.most_common(8):
        pct = count / total * 100
        table.add_row(issue, str(count), f"{pct:.0f}%")

    console.print(table)
    console.print()


def _print_failed(failed: list[tuple]) -> None:
    table = Table(
        title=f"[bold red]Failed Videos ({len(failed)})[/bold red]",
        show_lines=False,
    )
    table.add_column("Video", width=30)
    table.add_column("Operator", width=14)
    table.add_column("Error", width=40)

    for video, error in failed[:15]:
        table.add_row(
            video.filename,
            getattr(video, "operator_id", "—"),
            f"[red]{str(error)[:80]}[/red]",
        )

    if len(failed) > 15:
        table.add_row("", "", f"[dim]… and {len(failed) - 15} more[/dim]")

    console.print(table)
    console.print()


def _print_metric_overview(scores: list[VideoScore]) -> None:
    if not scores:
        return
    metric_names = sorted({k for s in scores for k in s.metric_scores})
    if not metric_names:
        return

    table = Table(title="Metric Overview", show_lines=False, row_styles=["", "dim"])
    table.add_column("Metric", width=28)
    table.add_column("Avg", justify="right", width=8)
    table.add_column("Min", justify="right", width=8)
    table.add_column("Max", justify="right", width=8)

    for metric in metric_names:
        vals = [s.metric_scores[metric] for s in scores if metric in s.metric_scores]
        if not vals:
            continue
        avg_val = mean(vals)
        style = "green" if avg_val >= 70 else "yellow" if avg_val >= 40 else "red"
        table.add_row(
            metric,
            f"[{style}]{avg_val:.1f}[/{style}]",
            f"{min(vals):.1f}",
            f"{max(vals):.1f}",
        )

    console.print(table)
    console.print()


def _format_low_metrics(score: VideoScore, limit: int = 3) -> str:
    items = sorted(score.metric_scores.items(), key=lambda item: item[1])[:limit]
    return ", ".join(f"{k}={v:.1f}" for k, v in items) if items else "—"


def _print_video_table(title: str, rows: list[VideoScore]) -> None:
    table = Table(title=title, show_lines=False, row_styles=["", "dim"])
    table.add_column("Video", width=28)
    table.add_column("Operator", width=14)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Grade", justify="center", width=6)
    table.add_column("Weakest Metrics", width=42)
    table.add_column("Issue", width=22)
    for s in rows:
        table.add_row(
            s.filename[:28],
            s.operator_id[:14],
            f"{s.composite_score:.1f}",
            _styled_grade(s.grade),
            _format_low_metrics(s),
            s.worst_issue if s.worst_issue != "none" else "[dim]—[/dim]",
        )
    console.print(table)
    console.print()


def _print_video_insights(scores: list[VideoScore]) -> None:
    if not scores:
        return
    sorted_scores = sorted(scores, key=lambda s: s.composite_score, reverse=True)
    _print_video_table("Top Videos", sorted_scores[:10])
    if len(sorted_scores) > 10:
        _print_video_table("Bottom Videos", sorted_scores[-10:])


def _print_operator_metric_breakdown(scores: list[VideoScore], rankings: list[dict]) -> None:
    if not scores or not rankings:
        return

    metric_names = sorted({k for s in scores for k in s.metric_scores})
    if not metric_names:
        return
    selected_metrics = metric_names[:4]

    by_operator: dict[str, list[VideoScore]] = {}
    for s in scores:
        by_operator.setdefault(s.operator_id, []).append(s)

    table = Table(title="Operator Metric Breakdown", show_lines=False, row_styles=["", "dim"])
    table.add_column("Operator", width=14)
    table.add_column("Final", justify="right", width=7)
    table.add_column("Grade", justify="center", width=6)
    for metric in selected_metrics:
        table.add_column(metric[:20], justify="right", width=9)

    for r in rankings[:10]:
        op_id = r["operator_id"]
        op_scores = by_operator.get(op_id, [])
        vals = {}
        for metric in selected_metrics:
            v = [s.metric_scores[metric] for s in op_scores if metric in s.metric_scores]
            vals[metric] = mean(v) if v else 0.0

        row = [op_id, f"{r['final_score']:.1f}", _styled_grade(r["grade"])]
        row.extend(f"{vals[m]:.1f}" for m in selected_metrics)
        table.add_row(*row)

    console.print(table)
    console.print()


def print_single_scorecard(score: VideoScore) -> None:
    """Print a compact scorecard for single-video mode."""
    console.print()

    # Score header
    grade_style = GRADE_STYLES.get(score.grade, "")
    console.print(
        Panel(
            f"[bold]{score.composite_score:.1f}[/bold]  [{grade_style}]{score.grade}[/{grade_style}]"
            f"  —  {grade_description(score.grade)}",
            title=f"[bold]{score.filename}[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Top 3 strongest and weakest metrics
    sorted_metrics = sorted(score.metric_scores.items(), key=lambda x: x[1])
    weakest = sorted_metrics[:3]
    strongest = sorted_metrics[-3:][::-1]

    cols = []
    weak_lines = ["[bold red]Weakest[/bold red]"]
    for k, v in weakest:
        weak_lines.append(f"  {k}: {v:.1f}")
    cols.append("\n".join(weak_lines))

    strong_lines = ["[bold green]Strongest[/bold green]"]
    for k, v in strongest:
        strong_lines.append(f"  {k}: {v:.1f}")
    cols.append("\n".join(strong_lines))

    console.print(Columns(cols, padding=(0, 4)))

    if score.worst_issue != "none":
        console.print(f"\n  Primary issue: [bold]{score.worst_issue}[/bold]")
    console.print()
