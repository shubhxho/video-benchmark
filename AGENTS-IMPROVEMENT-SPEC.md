# AGENTS-IMPROVEMENT-SPEC.md

Concrete improvements identified after auditing AGENTS.md, CLAUDE.md, skill files, and the codebase.

---

## Audit Summary

### What's good

- `CLAUDE.md` covers the minimal essentials: runtime, package manager, and the three most-used commands.
- `pyproject.toml` is well-structured: ruff, mypy strict, pytest paths, dev/llm dependency groups.
- `tests/conftest.py` provides reusable frame fixtures that keep unit tests free of real video files.
- `.gitignore` correctly excludes `videos/` and `.venv`.
- The `find-skills` skill in `.agents/skills/` is well-written and self-contained.

### What's missing

1. **AGENTS.md did not exist.** Agents had no repo-level guidance beyond the minimal CLAUDE.md stub.
2. **No lint/type-check commands** were documented — agents couldn't know `ruff` and `mypy` are available.
3. **No test command** was documented in CLAUDE.md (only run/add/sync).
4. **No web app commands** were documented anywhere.
5. **No environment variable reference** — `VB_*` prefix and `GEMINI_API_KEY` were undiscoverable.
6. **No guidance on optional ML dependencies** — agents could import heavy packages unconditionally.
7. **No guidance on where to add new metrics or CLI commands** — the extension points are non-obvious.
8. **`main.py` stub is misleading** — it prints "Hello from benchmark!" but the real entry point is the `benchmark` script in `pyproject.toml`. No documentation clarified this.
9. **`.env` file inside `src/video_benchmark/`** is not in `.gitignore` and could leak secrets.
10. **`CLAUDE.md` states Python 3.14+** but `pyproject.toml` requires `>=3.12` and `.python-version` likely pins 3.12/3.13. This is a factual error.

### What's wrong

- **CLAUDE.md Python version claim is incorrect**: says "Python 3.14+" but `pyproject.toml` says `>=3.12`. Agents following CLAUDE.md would assume a version that doesn't exist yet.
- **README.md contains absolute local paths** (e.g. `/Users/shubh/Documents/video-benchmark/results`) — these are developer-machine artifacts that break in any other environment and mislead agents reading the README.
- **`src/video_benchmark/.env` is tracked** (or at least present) and not gitignored — a security risk.

---

## Improvement Spec

### 1. Fix CLAUDE.md Python version

**File**: `CLAUDE.md`  
**Change**: Replace "Python 3.14+" with "Python 3.12+".  
**Why**: The incorrect version causes agents to make wrong assumptions about language features and available packages.

```markdown
## Project Overview

Benchmark project managed with `uv`. Python 3.12+.
```

### 2. Expand CLAUDE.md with test and lint commands

**File**: `CLAUDE.md`  
**Change**: Add test, lint, and type-check commands so agents can verify changes without reading pyproject.toml.

```markdown
## Commands

- **Run**: `uv run main.py`
- **CLI**: `uv run benchmark --help`
- **Test**: `uv run pytest`
- **Lint**: `uv run ruff check src tests`
- **Type-check**: `uv run mypy src`
- **Add dependency**: `uv add <package>`
- **Sync environment**: `uv sync`
```

### 3. Gitignore the `.env` file inside the package

**File**: `.gitignore`  
**Change**: Add `src/video_benchmark/.env` (or `**/.env`) to prevent accidental secret commits.

```gitignore
# Environment secrets
**/.env
.env
```

### 4. Fix absolute paths in README.md

**File**: `README.md`  
**Change**: Replace all occurrences of `/Users/shubh/Documents/video-benchmark/` with relative paths or remove the path prefix entirely.

Examples:
- `[results/](/Users/shubh/Documents/video-benchmark/results)` → `[results/](results/)`
- `[web/](/Users/shubh/Documents/video-benchmark/web)` → `[web/](web/)`
- All file links in "How The Web App Works" and "How The Python Pipeline Works" sections.

### 5. Clarify the `main.py` stub

**File**: `main.py` (or `AGENTS.md`)  
**Change**: Add a one-line comment to `main.py` explaining it is a placeholder, and document this in AGENTS.md (already done in the new AGENTS.md created in this session).

```python
# Stub entry point. The real CLI is `uv run benchmark` (src/video_benchmark/cli.py).
def main():
    print("Hello from benchmark!")
```

### 6. Add `.ona/automations.yaml` or document dev setup

**File**: `.devcontainer/devcontainer.json`  
**Change**: The devcontainer uses the 10 GB universal image. For a Python-only project, switch to `mcr.microsoft.com/devcontainers/python:3.12` and add a `postCreateCommand` to run `uv sync`, so agents and developers get a ready-to-use environment on container start.

```json
{
  "name": "video-benchmark",
  "image": "mcr.microsoft.com/devcontainers/python:3.12",
  "postCreateCommand": "pip install uv && uv sync"
}
```

### 7. Add `ffmpeg` to devcontainer features

**File**: `.devcontainer/devcontainer.json`  
**Change**: The CLI requires `ffmpeg` on `PATH`. The devcontainer does not install it. Add the feature so the environment is self-contained.

```json
{
  "features": {
    "ghcr.io/devcontainers/features/ffmpeg:1": {}
  }
}
```

### 8. Document the `find-skills` skill trigger in AGENTS.md

**File**: `AGENTS.md`  
**Change**: Add a section noting that `.agents/skills/find-skills/SKILL.md` exists and when agents should invoke it (when a user asks for a capability that might exist as an installable skill).

```markdown
## Skills

- **find-skills** (`.agents/skills/find-skills/SKILL.md`): use when the user asks for a capability
  that might exist as an installable skill. Run `npx skills find <query>` to search.
```

### 9. Add `results/` to `.gitignore` (or document the exception)

**File**: `.gitignore`  
**Change**: The `results/` directory contains generated artifacts. Either add it to `.gitignore` with a note that sample artifacts are intentionally committed via `git add -f`, or add a `.gitkeep` and exclude the generated files by pattern.

```gitignore
# Generated results (sample artifacts committed explicitly)
results/*.csv
results/*.json
results/*.html
```

---

## Priority Order

| Priority | Item | Effort |
|---|---|---|
| High | Fix Python version in CLAUDE.md (#1) | Trivial |
| High | Gitignore `.env` (#3) | Trivial |
| High | Fix absolute paths in README (#4) | Small |
| Medium | Expand CLAUDE.md commands (#2) | Trivial |
| Medium | Add ffmpeg to devcontainer (#7) | Small |
| Medium | Clarify `main.py` stub (#5) | Trivial |
| Low | Switch devcontainer image (#6) | Small |
| Low | Document find-skills in AGENTS.md (#8) | Trivial |
| Low | Gitignore results patterns (#9) | Trivial |
