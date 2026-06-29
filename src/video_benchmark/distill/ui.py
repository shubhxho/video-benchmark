"""A charm.sh-inspired terminal theme for the distillation report.

Tasteful, restrained, high-contrast: rounded panels, a muted palette with a
single pink accent (lipgloss vibes), inline gauge bars, and clear hierarchy.
"""

from __future__ import annotations

import math

from rich.align import Align
from rich.box import ROUNDED
from rich.console import RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# --- palette (charm/lipgloss-ish) -------------------------------------------
PINK = "#FF6AC1"
VIOLET = "#B794F6"
GREEN = "#74E39B"
AMBER = "#F6C177"
RED = "#FF6B81"
CYAN = "#67E8F9"
MUTED = "#7A7C90"
FAINT = "#55576a"

FILLED = "█"
EMPTY = "░"


def _corr_color(v: float) -> str:
    if v >= 0.8:
        return GREEN
    if v >= 0.5:
        return CYAN
    if v >= 0.3:
        return AMBER
    return RED


def gauge(value: float, width: int = 16, vmin: float = 0.0, vmax: float = 1.0) -> Text:
    """A colored fill bar for a value in [vmin, vmax]; dim 'n/a' if NaN."""
    if value is None or math.isnan(value):
        return Text("n/a", style=f"italic {FAINT}")
    frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    n = round(frac * width)
    color = _corr_color(value if vmax == 1.0 else frac)
    bar = Text(FILLED * n, style=color)
    bar.append(EMPTY * (width - n), style=FAINT)
    bar.append(f"  {value:+.2f}" if vmax == 1.0 else f"  {value:.0f}", style=color)
    return bar


def banner(title: str, subtitle: str) -> Panel:
    body = Text()
    body.append("◆ ", style=PINK)
    body.append(title, style=f"bold {PINK}")
    body.append("\n")
    body.append(subtitle, style=MUTED)
    return Panel(body, box=ROUNDED, border_style=PINK, padding=(1, 3))


def rule(text: str) -> Rule:
    return Rule(Text(f" {text} ", style=f"bold {VIOLET}"), style=FAINT, align="left")


def stat_card(label: str, value: str, sub: str, accent: str) -> Panel:
    body = Text()
    body.append(value, style=f"bold {accent}")
    body.append(f"\n{sub}", style=MUTED)
    return Panel(
        body,
        title=Text(label, style=f"bold {MUTED}"),
        box=ROUNDED,
        border_style=accent,
        padding=(1, 2),
        width=30,
    )


def badge(text: str, color: str) -> Text:
    return Text(f" {text} ", style=f"bold black on {color}")


def panelize(title: str, renderable: RenderableType, accent: str = VIOLET) -> Panel:
    return Panel(
        renderable,
        title=Text(title, style=f"bold {accent}"),
        title_align="left",
        box=ROUNDED,
        border_style=FAINT,
        padding=(1, 2),
    )


def clean_table() -> Table:
    """A borderless table for use inside a panel."""
    t = Table(box=None, show_edge=False, pad_edge=False, expand=True)
    t.header_style = f"bold {MUTED}"
    return t


def headline_cards(cards: list[Panel]) -> RenderableType:
    grid = Table.grid(padding=(0, 2))
    grid.add_row(*cards)
    return Align.center(grid)
