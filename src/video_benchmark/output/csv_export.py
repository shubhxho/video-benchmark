"""Export operator rankings to CSV."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from video_benchmark.scoring.scorer import VideoScore


def export_rankings_csv(rankings: list[dict], output_dir: Path) -> Path:
    """Export operator rankings to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rankings.csv"

    columns = [
        "rank", "operator_id", "final_score", "grade",
        "mean_score", "consistency_bonus", "video_count",
        "usable_pct", "worst_issue",
    ]
    rows = [{k: r[k] for k in columns} for r in rankings]

    df = pl.DataFrame(rows)
    df.write_csv(output_path)
    return output_path


def export_video_scores_csv(scores: list[VideoScore], output_dir: Path) -> Path:
    """Export per-video explicit scores/metrics to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "video_scores.csv"

    rows: list[dict] = []
    for s in sorted(scores, key=lambda x: x.composite_score, reverse=True):
        row: dict[str, str | float | int] = {
            "operator_id": s.operator_id,
            "filename": s.filename,
            "video_path": s.video_path,
            "composite_score": s.composite_score,
            "grade": s.grade,
            "worst_issue": s.worst_issue,
        }
        for k, v in s.metric_scores.items():
            row[f"metric_{k}"] = v
        rows.append(row)

    if rows:
        pl.DataFrame(rows).write_csv(output_path)
    else:
        pl.DataFrame(
            {
                "operator_id": [],
                "filename": [],
                "video_path": [],
                "composite_score": [],
                "grade": [],
                "worst_issue": [],
            }
        ).write_csv(output_path)
    return output_path
