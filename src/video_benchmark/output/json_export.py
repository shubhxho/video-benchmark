"""Export detailed results to JSON."""

from __future__ import annotations

import json
from pathlib import Path

from video_benchmark.pipeline.orchestrator import FailedVideo, RunInfoValue
from video_benchmark.scoring.aggregator import OperatorRanking
from video_benchmark.scoring.scorer import VideoScore


def export_detailed_json(
    scores: list[VideoScore],
    rankings: list[OperatorRanking],
    failed: list[FailedVideo],
    output_dir: Path,
    run_info: dict[str, RunInfoValue] | None = None,
) -> Path:
    """Export detailed per-video results and rankings to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detailed_results.json"

    data = {
        "summary": {
            "total_videos": len(scores) + len(failed),
            "scored_videos": len(scores),
            "failed_videos": len(failed),
            "operators": len(rankings),
        },
        "run_info": run_info or {},
        "operator_rankings": rankings,
        "video_scores": [
            {
                "operator_id": s.operator_id,
                "filename": s.filename,
                "composite_score": s.composite_score,
                "grade": s.grade,
                "worst_issue": s.worst_issue,
                "recommendations": s.recommendations,
                "scoring_notes": s.scoring_notes,
                "metric_scores": s.metric_scores,
                "metric_weights": s.metric_weights,
                "score_contributions": s.score_contributions,
                "raw_metrics": s.raw_metrics,
                "segment_scores": s.segment_scores,
            }
            for s in sorted(scores, key=lambda x: x.composite_score, reverse=True)
        ],
        "failed_videos": [
            {"filename": v.filename, "operator_id": v.operator_id, "error": err}
            for v, err in failed
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path
