# video-benchmark

Score and inspect first-person operator video quality.

This repo currently contains two related products:

- a Python CLI pipeline for batch scoring, ranking, exporting, and reporting on operator videos
- a React + WebGPU web app for local in-browser analysis with FFmpeg WASM frame extraction and hoverable frame inspection

## What It Does

The project is built around a simple question: is this operator video usable?

It scores footage using a mix of classical vision metrics, temporal checks, audio analysis, and hand visibility signals. The browser app keeps analysis local to the machine, and the CLI supports larger offline batch jobs.

## Repo Layout

```text
.
├── configs/                 Default weights and example manifests
├── results/                 Example output artifacts
├── src/video_benchmark/     Python package and CLI pipeline
├── tests/                   Python test suite
└── web/                     React/Vite browser app
```

## Main Features

### Web app

- local browser-side analysis
- FFmpeg WASM frame extraction
- WebGPU metrics for brightness, sharpness, blur, and stability
- MediaPipe hand detection
- audio and temporal analysis
- minimal review UI with per-frame hover previews
- adaptive sampling for longer videos

### Python CLI

- batch scoring for local folders or S3 sources
- `v1` classical scoring and `v2` ML-enhanced scoring
- segment-based sampling for long videos
- CSV, JSON, and HTML report export
- optional compression workflow
- configurable scoring weights

## Requirements

### Web app

- Bun
- a modern browser with WebGPU support
  Chrome and Edge work best today

### Python CLI

- Python 3.13 (the ML stack — numba via librosa/pyiqa — has no 3.14 wheels yet)
- `ffmpeg` installed on the system path
- optional GPU / ML dependencies depending on which scoring path you use

## Quick Start

### Web app

```bash
cd web
bun install
bun run dev
```

Then open the Vite URL in your browser and drop a video onto the page.

Build for production:

```bash
cd web
bun run build
```

Preview the production build:

```bash
cd web
bun run preview
```

### Deploy (Vercel)

The web app deploys to Vercel and nothing else. The root [vercel.json](./vercel.json)
builds `web/` with bun and serves `web/dist` as a static site. It also sets the
`Cross-Origin-Opener-Policy: same-origin` / `Cross-Origin-Embedder-Policy: require-corp`
headers on every response — this cross-origin isolation is what unlocks
`SharedArrayBuffer`, and therefore the multi-threaded FFmpeg core and MediaPipe's
threaded WASM. Without it the app still works, just on the slower single-threaded
paths.

Import the repo into Vercel (or run `vercel`) and no extra configuration is
needed. The dev and preview servers set the same isolation headers via
[vite.config.ts](./web/vite.config.ts).

### Python CLI

Using `uv`:

```bash
uv sync
uv run benchmark --help
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
benchmark --help
```

## CLI Usage

### Score a local directory

```bash
uv run benchmark score --source local --path ./videos --output results
```

### Score a single video

```bash
uv run benchmark score-single ./videos/example.mp4
```

### Generate an HTML report

```bash
uv run benchmark score --source local --path ./videos --report
```

### Use the ML-enhanced scorer

```bash
uv run benchmark score --source local --path ./videos --weights-version v2
```

### Compress source videos

```bash
uv run benchmark compress ./videos --output compressed --codec h265
```

### Interactive TUI

Launch the full-screen Textual app. It opens with an interactive, keyboard-driven
configuration form (huh-style — selects, text inputs, a GPU toggle, and a
multi-select for extras, with inline validation), then flows into a live
dashboard: results stream in as videos complete, with tabs for operator rankings,
metric overview, common issues, failures, and a perf benchmark, plus per-video
drill-down. Press `r` for a new run, `b` to benchmark, `q` to quit.

```bash
uv run benchmark tui                          # opens the form with defaults
uv run benchmark tui --source local --path ./videos   # pre-fills the form
```

### Performance benchmark

Measure per-stage throughput. The default synthetic mode needs no ffmpeg or
sample videos and directly reflects the underlying CV/ML library performance:

```bash
uv run benchmark bench                      # synthetic (ms/frame + frames/sec per stage)
uv run benchmark bench --frames 120         # more synthetic frames
uv run benchmark bench --path ./videos      # real pipeline (videos/sec, per-stage share)
uv run benchmark bench --json bench.json    # also dump results for comparison
```

## Outputs

The CLI writes artifacts into the output directory, typically `results/`.

Common files:

- `rankings.csv`
- `video_scores.csv`
- `detailed_results.json`
- `report.html`

The repo already includes sample artifacts in [results/](./results).

## How The Web App Works

The browser app lives in [web/](./web).

High-level flow:

1. load the video
2. extract frames with FFmpeg WASM (multi-threaded core when the page is
   cross-origin isolated, single-threaded otherwise)
3. compute brightness, sharpness, blur, and optical-flow stability on WebGPU
4. run MediaPipe segmentation and holistic body/hand detection — each in its own
   Web Worker so they run concurrently with the WebGPU metrics on the main thread
5. analyze audio and temporal consistency
6. score the clip and render a compact review UI

Both heavy ML models and the WebGPU metrics run in parallel per frame, so the
per-frame cost is roughly the slowest single stage rather than their sum.

Relevant files:

- [App.tsx](./web/src/App.tsx)
- [frame-extractor.ts](./web/src/video/frame-extractor.ts) — single/multi-threaded ffmpeg selection
- [frame-previews.ts](./web/src/video/frame-previews.ts)
- [metrics-panel.tsx](./web/src/components/metrics-panel.tsx)
- [worker-rpc.ts](./web/src/metrics/worker-rpc.ts) — shared request/response client for the ML workers
- [body-mapping.ts](./web/src/metrics/body-mapping.ts) + [body-mapping-worker.ts](./web/src/metrics/body-mapping-worker.ts)
- [segmentation.ts](./web/src/metrics/segmentation.ts) + [segmentation-worker.ts](./web/src/metrics/segmentation-worker.ts)
- [brightness.ts](./web/src/gpu/brightness.ts)
- [sharpness.ts](./web/src/gpu/sharpness.ts)
- [blur.ts](./web/src/gpu/blur.ts)
- [optical-flow.ts](./web/src/gpu/optical-flow.ts)

When the page is cross-origin isolated the extraction step is labelled
"extracting frames (multithreaded)" so you can confirm the fast path is active.

## How The Python Pipeline Works

The Python package lives in [src/video_benchmark/](./src/video_benchmark).

High-level flow:

1. resolve videos from local disk, manifest, or S3
2. extract configured time segments
3. sample frames
4. compute per-frame and per-segment metrics
5. aggregate video scores and operator rankings
6. export results

Relevant files:

- [cli.py](./src/video_benchmark/cli.py)
- [orchestrator.py](./src/video_benchmark/pipeline/orchestrator.py)
- [config.py](./src/video_benchmark/config.py)

## Notes

- The web app runs analysis locally in the browser. Videos are not uploaded by the app itself.
- The web app depends on WebGPU availability for the GPU metric path.
- The Python CLI requires system `ffmpeg`.
- Some optional Python metrics depend on heavier ML packages such as `pyiqa`, `ultralytics`, `open-clip-torch`, and Torch optical flow support.
- The web build includes local FFmpeg core assets (both single- and multi-threaded), so the production bundle is intentionally larger than a typical small Vite app. Each core is fetched lazily, so a visitor only downloads the one their browser uses.
- The multi-threaded FFmpeg core is copied into `web/public/vendor/ffmpeg/` at build time by `scripts/copy-ffmpeg-core.mjs` (gitignored); `bun run dev`, `preview`, and `build` all run this copy step automatically.

## Development

Run the Python tests:

```bash
uv run pytest
```

Run the web app in development:

```bash
cd web
bun run dev
```

## Status

The browser app is the fastest way to try the project right now.

The Python CLI remains the better fit for:

- larger offline batches
- manifest-driven runs
- S3-backed inputs
- export-heavy workflows
