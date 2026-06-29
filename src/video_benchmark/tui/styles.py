"""Textual CSS for the benchmark TUI.

Kept as a Python constant (rather than a packaged .tcss file) so styling ships
with the wheel without extra package-data configuration.
"""

from __future__ import annotations

APP_CSS = """
/* ---------- Dashboard ---------- */
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

/* ---------- Config form (huh-style) ---------- */
#form {
    width: 80%;
    max-width: 90;
    margin: 1 2;
    padding: 1 2;
    border: round $primary;
    background: $surface;
}

#form-title {
    text-style: bold;
    color: $accent;
    text-align: center;
    width: 1fr;
}

#form-help {
    color: $text-muted;
    text-align: center;
    margin-bottom: 1;
}

.field-label {
    text-style: bold;
    margin-top: 1;
}

.field-desc {
    color: $text-muted;
    margin-bottom: 0;
}

.row {
    height: auto;
}

.row Input {
    width: 1fr;
    margin-right: 1;
}

.switch-row {
    height: 3;
    align-vertical: middle;
}

#gpu-label {
    margin-left: 1;
    color: $text-muted;
}

#extras {
    height: auto;
    max-height: 6;
    border: round $panel;
}

#form-error {
    height: auto;
    margin-top: 1;
}

#form-buttons {
    height: auto;
    margin-top: 1;
    align-horizontal: center;
}

#form-buttons Button {
    margin: 0 1;
}
"""
