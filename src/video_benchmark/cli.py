"""Typer CLI entrypoint for video-benchmark."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from video_benchmark.acceleration import detect_acceleration, require_ffmpeg
from video_benchmark.config import BenchmarkSettings, ScoringWeights, ScoringWeightsV2
from video_benchmark.compression import (
    CompressionPlan,
    compress_video,
    find_videos,
    human_size,
    probe_video,
    select_plan,
)
from video_benchmark.output.console import print_summary
from video_benchmark.output.csv_export import export_rankings_csv, export_video_scores_csv
from video_benchmark.output.html_report import export_html_report
from video_benchmark.output.json_export import export_detailed_json
from video_benchmark.pipeline.orchestrator import run_pipeline
from video_benchmark.sources.base import VideoFile
from video_benchmark.sources.local import LocalVideoSource
from video_benchmark.sources.manifest import load_manifest
from video_benchmark.sources.s3 import S3VideoSource

app = typer.Typer(
    name="benchmark",
    help="Score and rank operator video quality from headband-mounted cameras.",
    add_completion=False,
)
console = Console()


@app.command()
def score(
    source: Annotated[str, typer.Option(help="Video source: 'local' or 's3'")] = "local",
    path: Annotated[Path | None, typer.Option(help="Local video directory path")] = None,
    bucket: Annotated[str | None, typer.Option(help="S3 bucket name")] = None,
    prefix: Annotated[str, typer.Option(help="S3 key prefix")] = "",
    manifest: Annotated[Path | None, typer.Option(help="CSV manifest file path")] = None,
    output: Annotated[Path, typer.Option(help="Output directory")] = Path("results"),
    workers: Annotated[
        int,
        typer.Option(help="Parallel workers (<=0 auto-tunes, GPU-aware on Apple)"),
    ] = 0,
    sample_rate: Annotated[int, typer.Option(help="Frames per second to sample")] = 1,
    segments: Annotated[int, typer.Option(help="Number of segments to sample")] = 3,
    no_gpu: Annotated[bool, typer.Option("--no-gpu", help="Disable GPU acceleration")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Verbose logging")] = False,
    format: Annotated[str, typer.Option(help="Output format: csv, json, or both")] = "both",
    weights_version: Annotated[
        str,
        typer.Option(
            "--weights-version",
            help="Scoring model: v1 (classical) or v2 (ML-enhanced)",
        ),
    ] = "v2",
    weights_file: Annotated[
        Path | None,
        typer.Option(help="Custom V1 weights JSON file"),
    ] = None,
    weights_v2_file: Annotated[
        Path | None,
        typer.Option("--weights-v2-file", help="Custom V2 weights JSON file"),
    ] = None,
    report: Annotated[
        bool,
        typer.Option("--report", help="Generate HTML report with charts and frame thumbnails"),
    ] = False,
) -> None:
    """Score and rank videos from a directory or S3 bucket."""
    require_ffmpeg()

    if weights_version not in {"v1", "v2"}:
        console.print("[red]--weights-version must be 'v1' or 'v2'.[/red]")
        raise typer.Exit(1)

    scoring_weights = ScoringWeights()
    scoring_weights_v2 = ScoringWeightsV2()
    if weights_file:
        scoring_weights = ScoringWeights.from_json(weights_file)
    if weights_v2_file:
        scoring_weights_v2 = ScoringWeightsV2.from_json(weights_v2_file)

    settings = BenchmarkSettings(
        source=source,
        path=path,
        bucket=bucket,
        prefix=prefix,
        manifest=manifest,
        output=output,
        workers=workers,
        sample_rate=sample_rate,
        segments=segments,
        no_gpu=no_gpu,
        verbose=verbose,
        format=format,
        weights=scoring_weights,
        weights_version=weights_version,
        weights_v2=scoring_weights_v2,
        report=report,
    )

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # Resolve video list
    videos = _resolve_videos(settings)
    if not videos:
        console.print("[red]No videos found.[/red]")
        raise typer.Exit(1)

    console.print(f"Found [bold]{len(videos)}[/bold] videos to process.")

    accel = detect_acceleration(force_no_gpu=no_gpu)
    if accel.videotoolbox:
        console.print("[green]VideoToolbox hardware acceleration enabled.[/green]")

    start = time.time()
    result = run_pipeline(videos, settings)
    elapsed = time.time() - start

    # Output
    print_summary(
        result.scores,
        result.operator_rankings,
        result.failed,
        elapsed,
        run_info=result.run_info,
    )

    if format in ("csv", "both"):
        csv_path = export_rankings_csv(result.operator_rankings, output)
        console.print(f"Rankings CSV: [bold]{csv_path}[/bold]")
        video_csv_path = export_video_scores_csv(result.scores, output)
        console.print(f"Video Metrics CSV: [bold]{video_csv_path}[/bold]")

    if format in ("json", "both"):
        json_path = export_detailed_json(
            result.scores, result.operator_rankings, result.failed, output
        )
        console.print(f"Detailed JSON: [bold]{json_path}[/bold]")

    if report:
        report_path = export_html_report(
            result.scores,
            result.operator_rankings,
            result.failed,
            output,
            frame_cache=result.frame_cache,
            elapsed=elapsed,
        )
        console.print(f"HTML Report: [bold]{report_path}[/bold]")


@app.command(name="score-single")
def score_single(
    video_path: Annotated[Path, typer.Argument(help="Path to a single video file")],
    no_gpu: Annotated[bool, typer.Option("--no-gpu", help="Disable GPU acceleration")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Verbose logging")] = False,
    weights_version: Annotated[
        str,
        typer.Option(
            "--weights-version",
            help="Scoring model: v1 (classical) or v2 (ML-enhanced)",
        ),
    ] = "v2",
) -> None:
    """Quick-test a single video file."""
    require_ffmpeg()

    if not video_path.exists():
        console.print(f"[red]Video not found: {video_path}[/red]")
        raise typer.Exit(1)
    if weights_version not in {"v1", "v2"}:
        console.print("[red]--weights-version must be 'v1' or 'v2'.[/red]")
        raise typer.Exit(1)

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    video = VideoFile(
        operator_id=video_path.parent.name,
        video_path=str(video_path),
        filename=video_path.name,
    )

    settings = BenchmarkSettings(
        workers=1,
        no_gpu=no_gpu,
        verbose=verbose,
        weights_version=weights_version,
    )

    start = time.time()
    result = run_pipeline([video], settings)
    elapsed = time.time() - start

    print_summary(
        result.scores,
        result.operator_rankings,
        result.failed,
        elapsed,
        run_info=result.run_info,
    )


@app.command()
def compress(
    source: Annotated[Path, typer.Argument(help="Path to a video file or directory containing .mp4")],
    output: Annotated[Path, typer.Option(help="Output directory for compressed files")] = Path(
        "compressed"
    ),
    codec: Annotated[
        str,
        typer.Option(
            help="Codec to use: h264 (fast), h265 (balanced), av1 (smaller but slower)",
            case_sensitive=False,
        ),
    ] = "h265",
    crf: Annotated[int | None, typer.Option(help="Override CRF; lower=better quality")] = None,
    scale: Annotated[str | None, typer.Option(help="Optional scale filter, e.g. 1280:-2")] = None,
    audio_bitrate: Annotated[str, typer.Option(help="Audio bitrate (e.g. 96k, 128k)")] = "96k",
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing outputs")] = False,
    llm: Annotated[
        bool,
        typer.Option("--llm", help="Ask Gemini to refine compression settings"),
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="GEMINI_API_KEY", help="Gemini API key"),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Verbose ffmpeg output")] = False,
) -> None:
    """Compress videos with sensible defaults and optional LLM tuning."""
    require_ffmpeg()

    videos = list(find_videos(source))
    if not videos:
        console.print("[red]No .mp4 files found.[/red]")
        raise typer.Exit(1)

    accel = detect_acceleration()
    if accel.videotoolbox:
        console.print("[green]Using VideoToolbox hardware acceleration when available.[/green]")

    results: list[str] = []
    for vid in videos:
        probe = probe_video(vid)
        plan: CompressionPlan = select_plan(
            probe,
            codec=codec,
            crf=crf,
            scale=scale,
            use_llm=llm,
            api_key=api_key,
        )
        plan.audio_bitrate = audio_bitrate

        try:
            res = compress_video(
                vid,
                output_dir=output,
                plan=plan,
                accel=accel,
                overwrite=overwrite,
                verbose=verbose,
            )
            results.append(
                f"{vid.name}: {human_size(res.source_size)} -> {human_size(res.output_size)} "
                f"({res.ratio:.2f}x)"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed: {vid.name} ({exc})[/red]")

    if results:
        console.print("\n[bold]Compression results[/bold]")
        for line in results:
            console.print(line)


def _resolve_videos(settings: BenchmarkSettings) -> list[VideoFile]:
    """Resolve video list from settings."""
    if settings.manifest:
        return load_manifest(settings.manifest, settings.path)

    if settings.source == "s3":
        if not settings.bucket:
            console.print("[red]--bucket required for S3 source.[/red]")
            raise typer.Exit(1)
        src = S3VideoSource(settings.bucket, settings.prefix)
        return src.list_videos()

    if not settings.path:
        console.print("[red]--path required for local source.[/red]")
        raise typer.Exit(1)
    src = LocalVideoSource(settings.path)
    return src.list_videos()


if __name__ == "__main__":
    app()
