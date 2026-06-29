"""Convenience entrypoint so `uv run main.py` launches the CLI.

The real entrypoint is the `benchmark` console script (see pyproject.toml),
which maps to `video_benchmark.cli:app`.
"""

from __future__ import annotations

from video_benchmark.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
