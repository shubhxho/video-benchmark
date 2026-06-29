"""Export operator rankings to CSV."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from video_benchmark.scoring.aggregator import OperatorRanking
from video_benchmark.scoring.scorer import VideoScore


def export_rankings_csv(rankings: list[OperatorRanking], output_dir: Path) -> Path:
    """Export operator rankings to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rankings.csv"

    rows = [
        {
            "rank": r.get("rank", 0),
            "operator_id": r["operator_id"],
            "final_score": r["final_score"],
            "grade": r["grade"],
            "mean_score": r["mean_score"],
            "consistency_bonus": r["consistency_bonus"],
            "video_count": r["video_count"],
            "usable_pct": r["usable_pct"],
            "worst_issue": r["worst_issue"],
        }
        for r in rankings
    ]

    df = pl.DataFrame(rows)
    df.write_csv(output_path)
    return output_path


def export_video_scores_csv(scores: list[VideoScore], output_dir: Path) -> Path:
    """Export per-video explicit scores/metrics to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "video_scores.csv"

    rows: list[dict[str, str | float | int]] = []
    for s in sorted(scores, key=lambda x: x.composite_score, reverse=True):
        row: dict[str, str | float | int] = {
            "operator_id": s.operator_id,
            "filename": s.filename,
            "video_path": s.video_path,
            "composite_score": s.composite_score,
            "grade": s.grade,
            "worst_issue": s.worst_issue,
            "recommendations": " | ".join(s.recommendations),
            "scoring_notes": " | ".join(s.scoring_notes),
        }
        for k, v in s.metric_scores.items():
            row[f"metric_{k}"] = v
        for k, v in s.metric_weights.items():
            row[f"weight_{k}"] = v
        for k, v in s.score_contributions.items():
            row[f"contribution_{k}"] = v
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
                "recommendations": [],
                "scoring_notes": [],
            }
        ).write_csv(output_path)
    return output_path
