# AGENTS.md

Agent guidance for the `video-benchmark` repository.

## Project Overview

Two products in one repo:

- **Python CLI** (`src/video_benchmark/`) — batch scoring, ranking, and reporting for operator videos. Entry point: `uv run benchmark --help`.
- **Web app** (`web/`) — React + Vite + WebGPU browser-side analysis with FFmpeg WASM frame extraction.

## Repository Layout

```
configs/                  Default weights JSON and example manifest CSV
results/                  Sample output artifacts (CSV, JSON, HTML)
src/video_benchmark/      Python package
  cli.py                  Typer CLI entrypoint
  config.py               Settings, ScoringWeights v1/v2, BenchmarkSettings
  pipeline/               Orchestrator, frame sampler, segment extractor
  metrics/                Per-metric modules (brightness, sharpness, blur, …)
  scoring/                Score aggregation
  sources/                Local, S3, manifest video resolvers
  output/                 CSV, JSON, HTML exporters
  compression.py          FFmpeg compression workflow
  acceleration.py         GPU/VideoToolbox detection
  llm/                    Optional Gemini-based compression tuning
tests/                    pytest suite
web/                      React/Vite browser app (Bun)
  src/gpu/                WebGPU metric shaders
  src/video/              FFmpeg WASM frame extraction
  src/components/         UI components
```

## Commands

### Python CLI

```bash
uv sync                          # install / sync dependencies
uv run benchmark --help          # list all commands
uv run benchmark score --source local --path ./videos --output results
uv run benchmark score-single ./videos/example.mp4
uv run benchmark compress ./videos --output compressed --codec h265
uv run pytest                    # run tests
uv run ruff check src tests      # lint
uv run mypy src                  # type-check
```

### Web app

```bash
cd web
bun install
bun run dev      # development server
bun run build    # production build
bun run preview  # preview production build
```

## Key Conventions

- **Python version**: 3.12+ (pyproject.toml), `.python-version` pins the exact version.
- **Package manager**: `uv` for Python, `bun` for the web app.
- **Linting**: `ruff` with `E, F, I, N, UP, B, SIM` rules, line length 100.
- **Type checking**: `mypy` in strict mode.
- **Testing**: `pytest`, test files in `tests/`, fixtures in `tests/conftest.py`.
- **Settings**: `BenchmarkSettings` uses `pydantic-settings` with `VB_` env prefix.
- **Scoring models**: `v1` = classical CV metrics; `v2` = ML-enhanced (pyiqa, ultralytics, open-clip, torch).
- **Optional ML deps**: `pyiqa`, `ultralytics`, `open-clip-torch`, `torchvision` — heavy; guard with `detect_available_models()` before importing.
- **ffmpeg**: must be on `PATH` for the CLI; checked at startup via `require_ffmpeg()`.

## Environment Variables

| Variable | Purpose |
|---|---|
| `VB_*` | Any `BenchmarkSettings` field (e.g. `VB_WORKERS`, `VB_NO_GPU`) |
| `GEMINI_API_KEY` | Required for `--llm` compression tuning |
| `AWS_*` | Standard boto3 credentials for S3 source |

## Agent Guidance

- **Before editing Python**: run `uv run ruff check` and `uv run mypy src` to understand the current lint/type state.
- **Adding a metric**: create a module in `src/video_benchmark/metrics/`, inherit from `base.py`, register it in the orchestrator.
- **Adding a CLI command**: add a `@app.command()` function in `cli.py`; follow the existing `Annotated` + `typer.Option` pattern.
- **Adding a dependency**: use `uv add <package>` (or `uv add --group dev <package>` for dev-only).
- **Tests**: add tests in `tests/`; use fixtures from `conftest.py` for frame data; avoid real video files in unit tests.
- **Web changes**: the web app is independent of the Python package; changes in `web/` do not affect CLI tests.
- **Do not commit**: `src/video_benchmark/.env`, `videos/`, `results/` (sample artifacts already committed are intentional).
- **main.py**: the root `main.py` is a stub (`print("Hello from benchmark!")`); the real entry point is `src/video_benchmark/cli.py` via the `benchmark` script.
