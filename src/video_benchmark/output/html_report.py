"""HTML visual report generation with embedded charts and frame thumbnails."""

from __future__ import annotations

import base64
import io
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape

from video_benchmark.scoring.scorer import VideoScore


def _frame_to_b64_jpeg(frame: np.ndarray, max_width: int = 320) -> str:
    """Resize frame and encode as base64 JPEG."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(
            frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA
        )
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _make_grade_chart_b64(scores: list[VideoScore]) -> str | None:
    """Generate a grade distribution bar chart as base64 PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grade_counts = Counter(s.grade for s in scores)
        grades = ["A", "B", "C", "D", "F"]
        counts = [grade_counts.get(g, 0) for g in grades]
        colors = ["#22c55e", "#3b82f6", "#eab308", "#ef4444", "#dc2626"]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(grades, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Videos")
        ax.set_title("Grade Distribution")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except ImportError:
        return None


def export_html_report(
    scores: list[VideoScore],
    rankings: list[dict],
    failed: list[tuple],
    output_dir: Path,
    frame_cache: dict[str, dict[str, np.ndarray]] | None = None,
    elapsed: float = 0.0,
) -> Path:
    """Generate a self-contained HTML report.

    Args:
        scores: List of VideoScore results.
        rankings: Operator ranking dicts.
        failed: List of (VideoFile, error_msg) tuples.
        output_dir: Directory to write report to.
        frame_cache: Optional dict of {filename: {"best": frame, "worst": frame}}.
        elapsed: Processing time in seconds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.html"

    env = Environment(
        loader=PackageLoader("video_benchmark", "templates"),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    # Prepare frame cache as base64
    b64_cache: dict[str, dict[str, str]] = {}
    if frame_cache:
        for filename, frames in frame_cache.items():
            b64_cache[filename] = {}
            if "best" in frames and frames["best"] is not None:
                b64_cache[filename]["best"] = _frame_to_b64_jpeg(frames["best"])
            if "worst" in frames and frames["worst"] is not None:
                b64_cache[filename]["worst"] = _frame_to_b64_jpeg(
                    frames["worst"]
                )

    # Grade distribution
    grade_dist = Counter(s.grade for s in scores)
    grade_chart = _make_grade_chart_b64(scores)

    # Common issues
    issues = [s.worst_issue for s in scores if s.worst_issue != "none"]
    common_issues = Counter(issues).most_common(10)

    # Failed videos as simple objects
    failed_objs = [
        {"filename": v.filename, "error": err} for v, err in failed
    ]

    # Sort scores by composite_score desc
    sorted_scores = sorted(scores, key=lambda s: s.composite_score, reverse=True)

    html = template.render(
        total_videos=len(scores) + len(failed),
        scored_videos=len(scores),
        failed_count=len(failed),
        operator_count=len(rankings),
        elapsed=elapsed,
        grade_dist=grade_dist,
        grade_chart_b64=grade_chart,
        rankings=rankings,
        scores=sorted_scores,
        frame_cache=b64_cache,
        common_issues=common_issues,
        failed_videos=failed_objs,
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path
