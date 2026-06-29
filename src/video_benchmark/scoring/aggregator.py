"""Operator-level aggregation — average scores + consistency bonus."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import NotRequired, TypedDict

from video_benchmark.scoring.grader import assign_grade
from video_benchmark.scoring.scorer import VideoScore


class OperatorRanking(TypedDict):
    operator_id: str
    final_score: float
    grade: str
    mean_score: float
    consistency_bonus: float
    video_count: int
    usable_pct: str
    worst_issue: str
    rank: NotRequired[int]


def aggregate_operators(scores: list[VideoScore]) -> list[OperatorRanking]:
    """Aggregate video scores per operator, compute mean + consistency bonus."""
    by_operator: dict[str, list[VideoScore]] = defaultdict(list)
    for s in scores:
        by_operator[s.operator_id].append(s)

    rankings: list[OperatorRanking] = []

    for op_id, op_scores in by_operator.items():
        raw_scores = [s.composite_score for s in op_scores]
        mean_score = statistics.mean(raw_scores)

        # Consistency bonus: low stdev → bonus up to 3 points
        consistency_bonus = 0.0
        if len(raw_scores) > 1:
            stdev = statistics.stdev(raw_scores)
            # stdev < 5 gets full bonus, stdev > 20 gets no bonus
            consistency_bonus = max(0.0, min(3.0, (20 - stdev) / 5.0))

        final_score = min(100.0, mean_score + consistency_bonus)

        usable_count = sum(1 for s in op_scores if s.grade in ("A", "B"))
        usable_pct = usable_count / len(op_scores) * 100

        # Find worst issue across all videos
        issues = [s.worst_issue for s in op_scores if s.worst_issue != "none"]
        worst_issue = max(set(issues), key=issues.count) if issues else "none"

        rankings.append({
            "operator_id": op_id,
            "final_score": round(final_score, 1),
            "grade": assign_grade(final_score),
            "mean_score": round(mean_score, 1),
            "consistency_bonus": round(consistency_bonus, 1),
            "video_count": len(op_scores),
            "usable_pct": f"{usable_pct:.0f}%",
            "worst_issue": worst_issue,
        })

    rankings.sort(key=lambda r: r["final_score"], reverse=True)

    for i, r in enumerate(rankings, 1):
        r["rank"] = i

    return rankings
