"""Rich console output — summary tables and grade distribution."""

from __future__ import annotations

from collections import Counter
from statistics import mean

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from video_benchmark.scoring.grader import grade_description
from video_benchmark.scoring.scorer import VideoScore

console = Console()


def print_summary(
    scores: list[VideoScore],
    rankings: list[dict],
    failed: list[tuple],
    elapsed_seconds: float,
    run_info: dict[str, str | int | bool] | None = None,
) -> None:
    """Print a rich summary of benchmark results."""
    console.print()
    console.print(Panel.fit("[bold]Video Benchmark Results[/bold]", style="blue"))
    console.print()

    # Stats
    total = len(scores) + len(failed)
    console.print(f"  Videos analyzed: [bold]{len(scores)}[/bold] / {total}")
    if failed:
        console.print(f"  Failed: [red]{len(failed)}[/red]")
    console.print(f"  Operators: [bold]{len(rankings)}[/bold]")
    console.print(f"  Time: [bold]{elapsed_seconds:.1f}s[/bold]")
    if run_info:
        mode = run_info.get("executor_mode", "unknown")
        workers = run_info.get("workers_used", "?")
        req = run_info.get("workers_requested", "?")
        vt = "yes" if run_info.get("videotoolbox") else "no"
        mps = "yes" if run_info.get("mps_available") else "no"
        wv = run_info.get("weights_version", "unknown")
        console.print(
            "  Run mode: "
            f"[bold]{mode}[/bold], workers [bold]{workers}[/bold] (requested {req}), "
            f"videotoolbox={vt}, mps={mps}, weights={wv}"
        )
    console.print()

    # Grade distribution
    _print_grade_distribution(scores)
    _print_metric_overview(scores)

    # Top 10 / Bottom 10 operators
    if rankings:
        _print_operator_table("Top Operators", rankings[:10])
        if len(rankings) > 10:
            _print_operator_table("Bottom Operators", rankings[-10:])

    _print_video_insights(scores)
    _print_operator_metric_breakdown(scores, rankings)

    # Common issues
    _print_common_issues(scores)

    # Failed videos
    if failed:
        _print_failed(failed)


def _print_grade_distribution(scores: list[VideoScore]) -> None:
    grade_counts = Counter(s.grade for s in scores)
    grade_colors = {"A": "green", "B": "blue", "C": "yellow", "D": "red", "F": "bright_red"}

    console.print("[bold]Grade Distribution[/bold]")
    total = len(scores) or 1
    for grade in ["A", "B", "C", "D", "F"]:
        count = grade_counts.get(grade, 0)
        pct = count / total * 100
        bar_len = int(pct / 2)
        color = grade_colors[grade]
        bar = f"[{color}]{'#' * bar_len}[/{color}]"
        desc = grade_description(grade)
        console.print(f"  {grade} {bar} {count} ({pct:.0f}%) — {desc}")
    console.print()


def _print_operator_table(title: str, rankings: list[dict]) -> None:
    table = Table(title=title, show_lines=False)
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Operator", width=15)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Grade", justify="center", width=6)
    table.add_column("Videos", justify="right", width=7)
    table.add_column("Usable", justify="right", width=7)
    table.add_column("Issue", width=20)

    grade_styles = {"A": "green", "B": "blue", "C": "yellow", "D": "red", "F": "bright_red"}

    for r in rankings:
        style = grade_styles.get(r["grade"], "")
        table.add_row(
            str(r["rank"]),
            r["operator_id"],
            f"{r['final_score']:.1f}",
            f"[{style}]{r['grade']}[/{style}]",
            str(r["video_count"]),
            r["usable_pct"],
            r["worst_issue"],
        )
    console.print(table)
    console.print()


def _print_common_issues(scores: list[VideoScore]) -> None:
    issues = [s.worst_issue for s in scores if s.worst_issue != "none"]
    if not issues:
        return
    counter = Counter(issues)
    console.print("[bold]Most Common Issues[/bold]")
    for issue, count in counter.most_common(5):
        console.print(f"  {issue}: {count} videos")
    console.print()


def _print_failed(failed: list[tuple]) -> None:
    console.print(f"[bold red]Failed Videos ({len(failed)})[/bold red]")
    for video, error in failed[:10]:
        console.print(f"  [red]{video.filename}[/red]: {error}")
    if len(failed) > 10:
        console.print(f"  ... and {len(failed) - 10} more")
    console.print()


def _print_metric_overview(scores: list[VideoScore]) -> None:
    if not scores:
        return
    metric_names = sorted({k for s in scores for k in s.metric_scores})
    if not metric_names:
        return

    table = Table(title="Metric Overview (All Videos)", show_lines=False)
    table.add_column("Metric", width=28)
    table.add_column("Avg", justify="right", width=8)
    table.add_column("Min", justify="right", width=8)
    table.add_column("Max", justify="right", width=8)

    for metric in metric_names:
        vals = [s.metric_scores[metric] for s in scores if metric in s.metric_scores]
        if not vals:
            continue
        table.add_row(metric, f"{mean(vals):.1f}", f"{min(vals):.1f}", f"{max(vals):.1f}")

    console.print(table)
    console.print()


def _format_low_metrics(score: VideoScore, limit: int = 3) -> str:
    items = sorted(score.metric_scores.items(), key=lambda item: item[1])[:limit]
    return ", ".join(f"{k}={v:.1f}" for k, v in items) if items else "-"


def _print_video_table(title: str, rows: list[VideoScore]) -> None:
    table = Table(title=title, show_lines=False)
    table.add_column("Video", width=28)
    table.add_column("Operator", width=14)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Grade", justify="center", width=6)
    table.add_column("Worst Metrics", width=42)
    table.add_column("Issue", width=20)
    for s in rows:
        table.add_row(
            s.filename,
            s.operator_id,
            f"{s.composite_score:.1f}",
            s.grade,
            _format_low_metrics(s),
            s.worst_issue,
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

    table = Table(title="Operator Metric Breakdown", show_lines=False)
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

        row = [op_id, f"{r['final_score']:.1f}", r["grade"]]
        row.extend(f"{vals[m]:.1f}" for m in selected_metrics)
        table.add_row(*row)

    console.print(table)
    console.print()
