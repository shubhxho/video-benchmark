# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`video-benchmark` scores and ranks first-person operator video quality (e.g.
headband-mounted cameras). It has two parts:

- **Python CLI** (`src/video_benchmark/`, managed with `uv`, Python **3.13**) — batch
  scoring, ranking, exporting (CSV/JSON/HTML), an interactive Textual TUI, a
  performance benchmark, and a compression workflow.
- **Web app** (`web/`) — a separate React + WebGPU in-browser analyzer. Independent
  of the Python package; not covered here.

## Commands

All Python commands run through `uv`:

- **Install/sync**: `uv sync` (add `--group llm` for Gemini, `--group audio` for the
  librosa audio metric, `--group dev` for tooling)
- **Run the CLI**: `uv run benchmark --help` (or `uv run main.py`)
- **Score a directory**: `uv run benchmark score --source local --path ./videos`
- **Score one file**: `uv run benchmark score-single ./videos/clip.mp4`
- **Interactive TUI**: `uv run benchmark tui --source local --path ./videos`
- **Perf benchmark**: `uv run benchmark bench` (synthetic, no ffmpeg needed) or
  `uv run benchmark bench --path ./videos` (real pipeline); add `--json out.json`
- **Compress**: `uv run benchmark compress ./videos --codec h265`
- **Tests**: `uv run pytest`
- **Lint / types**: `uv run ruff check` and `uv run mypy src`
- **Add a dependency**: `uv add <package>`

Requires system `ffmpeg` on PATH for any path that decodes real video.

## Architecture

- `cli.py` — Typer entrypoint; commands: `score`, `score-single`, `compress`,
  `bench`, `tui`.
- `config.py` — `BenchmarkSettings` (pydantic-settings, `VB_` env prefix) plus the
  `ScoringWeights` / `ScoringWeightsV2` models (the single source of truth for
  weights) and `detect_available_models()`.
- `pipeline/orchestrator.py` — `run_pipeline()` drives per-video processing
  (`process_single_video`) serially or across a thread pool. UI-agnostic: it
  reports progress through a `ProgressReporter` and optionally collects per-stage
  timings (`settings.collect_timings`).
- `pipeline/reporting.py` — `ProgressReporter` protocol + `RichProgressReporter`
  (CLI default). The TUI supplies its own reporter.
- `metrics/` — individual metric implementations (brightness, sharpness, stability,
  hands, IQA, blur, anomalies, scene, temporal, depth, vqa, audio). Heavy ML
  metrics are imported lazily and degrade gracefully when their package is absent.
- `scoring/` — `score_video()` (v1 classical / v2 ML-enhanced) and operator
  aggregation.
- `output/` — Rich console summary, CSV/JSON exporters, HTML report.
- `benchmark/runner.py` — synthetic + real performance benchmarks.
- `tui/` — Textual dashboard (`BenchmarkApp`) that runs the pipeline in a worker
  thread and shows live results, summary tabs, drill-down, and a benchmark tab.

## Notes

- `numba` (via the optional `librosa` audio metric) has no Python 3.14 wheels yet,
  so `librosa` lives in the opt-in `audio` group. Everything else targets 3.14.
- Optional ML metrics (pyiqa, ultralytics, open-clip, torchvision RAFT, depth,
  DOVER) are detected at runtime; missing ones fall back to classical metrics.
