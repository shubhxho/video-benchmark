"""Textual CSS for the benchmark TUI.

Kept as a Python constant (rather than a packaged .tcss file) so styling ships
with the wheel without extra package-data configuration.
"""

from __future__ import annotations

APP_CSS = """
#runinfo {
    padding: 0 1;
    height: 1;
    color: $text-muted;
}

#pb {
    margin: 0 1 1 1;
}

TabbedContent {
    height: 1fr;
}

#live-table {
    height: 1fr;
}

#detail {
    height: auto;
    max-height: 16;
    border-top: solid $primary;
    padding: 1;
}

VerticalScroll {
    height: 1fr;
}
"""
