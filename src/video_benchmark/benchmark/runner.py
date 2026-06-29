"""Per-stage performance benchmark for the video-benchmark metric pipeline.

Two modes:

* **synthetic** — generate random frames and time each metric stage in isolation.
  Needs no ffmpeg and no sample videos, so it runs anywhere and directly measures
  the impact of the underlying CV/ML libraries.
* **real** — run the actual pipeline over a directory of videos with per-stage
  timing collection enabled, reporting throughput and per-stage share.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from video_benchmark.acceleration import detect_acceleration
from video_benchmark.config import BenchmarkSettings, detect_available_models

if TYPE_CHECKING:
    from video_benchmark.sources.base import VideoFile

# Libraries whose versions materially affect benchmark numbers.
_KEY_LIBS = [
    "opencv-python",
    "numpy",
    "scipy",
    "mediapipe",
    "pillow",
    "polars",
    "rich",
    "typer",
    "pydantic",
    "textual",
    "torch",
    "ultralytics",
    "open-clip-torch",
    "pyiqa",
]


@dataclass
class StageTiming:
    """Timing result for a single pipeline stage."""

    stage: str
    frames: int
    total_ms: float

    @property
    def ms_per_frame(self) -> float:
        return self.total_ms / self.frames if self.frames else 0.0

    @property
    def fps(self) -> float:
        return (self.frames / (self.total_ms / 1000.0)) if self.total_ms > 0 else 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark output for one run."""

    mode: str
    frames: int
    stages: list[StageTiming]
    env: dict[str, str]
    models: dict[str, bool]
    accel: dict[str, bool]
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frames": self.frames,
            "stages": [
                {
                    "stage": s.stage,
                    "frames": s.frames,
                    "total_ms": round(s.total_ms, 3),
                    "ms_per_frame": round(s.ms_per_frame, 4),
                    "fps": round(s.fps, 2),
                }
                for s in self.stages
            ],
            "env": self.env,
            "models": self.models,
            "accel": self.accel,
            "extra": {k: round(v, 4) for k, v in self.extra.items()},
        }


def build_env_info() -> dict[str, str]:
    """Collect Python + key library versions for the report header."""
    env = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "implementation": sys.implementation.name,
    }
    for lib in _KEY_LIBS:
        try:
            env[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            env[lib] = "—"
    return env


# --- Synthetic stage runners (each consumes a frame list, returns frame units) ---


def _bench_brightness(frames: list[np.ndarray]) -> int:
    from video_benchmark.metrics.brightness import BrightnessMetric

    m = BrightnessMetric()
    for f in frames:
        m.compute(f)
    return len(frames)


def _bench_sharpness(frames: list[np.ndarray]) -> int:
    from video_benchmark.metrics.sharpness import SharpnessMetric

    m = SharpnessMetric()
    for f in frames:
        m.compute(f)
    return len(frames)


def _bench_stability(frames: list[np.ndarray]) -> int:
    from video_benchmark.metrics.stability import StabilityMetric

    m = StabilityMetric()
    for i in range(1, len(frames)):
        m.compute_flow(frames[i - 1], frames[i])
    return max(0, len(frames) - 1)


def _bench_hand(frames: list[np.ndarray]) -> int:
    from video_benchmark.metrics.hand_detection import HandDetectionMetric

    hd = HandDetectionMetric()
    try:
        for f in frames:
            hd.detect(f)
    finally:
        hd.close()
    return len(frames)


def _bench_anomaly(frames: list[np.ndarray]) -> int:
    from video_benchmark.metrics.anomalies import AnomalyDetector

    det = AnomalyDetector()
    for f in frames:
        det.compute_anomaly_score(f)
        det.detect_anomalies(f)
    return len(frames)


def _bench_blur(frames: list[np.ndarray]) -> int:
    from video_benchmark.metrics.blur import BlurClassifier

    clf = BlurClassifier()
    for f in frames:
        clf.classify(f)
    return len(frames)


def _bench_temporal(frames: list[np.ndarray]) -> int:
    from video_benchmark.metrics.temporal import TemporalConsistencyMetric

    TemporalConsistencyMetric().compute(frames)
    return len(frames)


_CV_STAGES = [
    ("brightness", _bench_brightness),
    ("sharpness", _bench_sharpness),
    ("stability", _bench_stability),
    ("hand", _bench_hand),
    ("anomaly", _bench_anomaly),
    ("blur", _bench_blur),
    ("temporal", _bench_temporal),
]


def synthetic_benchmark(
    num_frames: int = 60,
    width: int = 640,
    height: int = 480,
    seed: int = 1234,
) -> BenchmarkReport:
    """Time each metric stage on synthetic random frames (no ffmpeg required)."""
    rng = np.random.default_rng(seed)
    frames = [
        rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]

    stages: list[StageTiming] = []
    for name, fn in _CV_STAGES:
        start = time.perf_counter()
        units = fn(frames)
        total_ms = (time.perf_counter() - start) * 1000.0
        stages.append(StageTiming(name, units, total_ms))

    accel = detect_acceleration()
    return BenchmarkReport(
        mode="synthetic",
        frames=num_frames,
        stages=stages,
        env=build_env_info(),
        models=detect_available_models(),
        accel={
            "videotoolbox": accel.videotoolbox,
            "mps_available": accel.mps_available,
            "ffmpeg": accel.ffmpeg_path is not None,
        },
        extra={"frame_width": float(width), "frame_height": float(height)},
    )


def real_benchmark(
    videos: list[VideoFile],
    settings: BenchmarkSettings,
) -> BenchmarkReport:
    """Run the actual pipeline with timing enabled and report throughput."""
    from video_benchmark.pipeline.orchestrator import run_pipeline
    from video_benchmark.pipeline.reporting import RichProgressReporter

    timed_settings = settings.model_copy(update={"collect_timings": True})

    start = time.perf_counter()
    result = run_pipeline(videos, timed_settings, reporter=RichProgressReporter())
    wall_s = time.perf_counter() - start

    total_frames = int(
        sum(s.raw_metrics.get("total_frames", 0) for s in result.scores)
    )
    stages = [
        StageTiming(stage, total_frames, secs * 1000.0)
        for stage, secs in sorted(
            result.timings.items(), key=lambda kv: kv[1], reverse=True
        )
    ]

    accel = detect_acceleration(force_no_gpu=settings.no_gpu)
    scored = len(result.scores)
    return BenchmarkReport(
        mode="real",
        frames=total_frames,
        stages=stages,
        env=build_env_info(),
        models=detect_available_models(),
        accel={
            "videotoolbox": accel.videotoolbox,
            "mps_available": accel.mps_available,
            "ffmpeg": accel.ffmpeg_path is not None,
        },
        extra={
            "wall_clock_s": wall_s,
            "videos_scored": float(scored),
            "videos_failed": float(len(result.failed)),
            "videos_per_sec": (scored / wall_s) if wall_s > 0 else 0.0,
            "frames_per_sec": (total_frames / wall_s) if wall_s > 0 else 0.0,
        },
    )


def render_report(report: BenchmarkReport, console: Console | None = None) -> None:
    """Render a benchmark report as Rich tables."""
    console = console or Console()

    # --- Environment header ---
    env_lines = [
        f"[bold]Python[/bold] {report.env.get('python', '?')} "
        f"({report.env.get('implementation', '?')})  "
        f"[bold]{report.env.get('platform', '?')}[/bold]",
    ]
    lib_bits = [
        f"{lib}={report.env[lib]}"
        for lib in _KEY_LIBS
        if report.env.get(lib, "—") != "—"
    ]
    env_lines.append("[dim]" + "  ".join(lib_bits) + "[/dim]")
    accel_bits = [f"{k}={'✓' if v else '✗'}" for k, v in report.accel.items()]
    active_models = [name for name, ok in report.models.items() if ok]
    env_lines.append("accel: " + "  ".join(accel_bits))
    env_lines.append(
        "ml models active: "
        + (", ".join(active_models) if active_models else "[dim]none[/dim]")
    )
    console.print(
        Panel(
            "\n".join(env_lines),
            title=f"[bold]Benchmark — {report.mode} mode[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # --- Per-stage table ---
    table = Table(title=f"Per-stage timing ({report.frames} frames)", show_lines=False)
    table.add_column("Stage", width=16)
    table.add_column("Frames", justify="right", width=8)
    table.add_column("Total ms", justify="right", width=12)
    table.add_column("ms/frame", justify="right", width=10)
    table.add_column("frames/sec", justify="right", width=12)
    if report.mode == "real":
        table.add_column("share", justify="right", width=8)

    total_ms = sum(s.total_ms for s in report.stages) or 1.0
    for s in report.stages:
        row = [
            s.stage,
            str(s.frames),
            f"{s.total_ms:.1f}",
            f"{s.ms_per_frame:.3f}",
            f"{s.fps:.1f}",
        ]
        if report.mode == "real":
            row.append(f"{s.total_ms / total_ms * 100:.0f}%")
        table.add_row(*row)
    console.print(table)

    if report.extra:
        extra_lines = []
        for key in (
            "wall_clock_s",
            "videos_scored",
            "videos_failed",
            "videos_per_sec",
            "frames_per_sec",
        ):
            if key in report.extra:
                extra_lines.append(f"{key.replace('_', ' ')}: [bold]{report.extra[key]:.2f}[/bold]")
        if extra_lines:
            console.print(Panel("\n".join(extra_lines), title="Throughput", border_style="green"))


def report_to_json(report: BenchmarkReport) -> str:
    import json

    return json.dumps(report.to_dict(), indent=2)


# Re-export for callers that want the raw dict
def report_as_dict(report: BenchmarkReport) -> dict[str, Any]:
    return asdict(report)
