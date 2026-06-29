"""Performance benchmarking for the metric pipeline."""

from __future__ import annotations

from video_benchmark.benchmark.runner import (
    BenchmarkReport,
    StageTiming,
    build_env_info,
    real_benchmark,
    render_report,
    synthetic_benchmark,
)

__all__ = [
    "BenchmarkReport",
    "StageTiming",
    "build_env_info",
    "real_benchmark",
    "render_report",
    "synthetic_benchmark",
]
