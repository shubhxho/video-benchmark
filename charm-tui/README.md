# charm-tui — Go front-end for the distillation pipeline

A [Bubble Tea](https://github.com/charmbracelet/bubbletea) +
[Lip Gloss](https://github.com/charmbracelet/lipgloss) (charm.sh) terminal UI for
the compact-quality-model distillation. It runs the Python pipeline, streams the
progress live, and renders the results natively in Go.

The ML stays in Python (`video_benchmark.distill`, which has no Go equivalent for
torch/pyiqa/MobileCLIP); this binary is the presentation/orchestration layer. The
two talk over a JSON contract: the Python command prints machine-readable results
to **stdout** (progress to **stderr**) when invoked with `--emit-json`.

## Build & run

```bash
cd charm-tui
go build -o vbtui .

# from the repo root (so it finds pyproject.toml + videos/)
./charm-tui/vbtui --videos videos --epochs 300 --fps 3
```

Keys: `q` / `esc` quit.

### Render a saved result (non-interactive)

```bash
# capture results once…
uv run python -m video_benchmark.distill --emit-json --videos videos > results.json
# …then render them anytime
./charm-tui/vbtui -from results.json
```

## Layout
- `main.go` — Bubble Tea model (running → done/error) + Lip Gloss rendering
- `run.go` — spawns the Python pipeline, streams stderr progress, parses stdout JSON
- `result.go` — the JSON contract (nullable correlations render as `n/a`)
- `styles.go` — the charm palette + panel/card/badge styles
