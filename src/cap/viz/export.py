"""Figure export — PDF (paper), SVG (editable), PNG (slides), HTML (web)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, path: str | Path, *, formats: list[str] | None = None) -> None:
    """Save figure once or in multiple formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if formats is None:
        fig.savefig(path)
    else:
        for fmt in formats:
            fig.savefig(path.with_suffix(f".{fmt}"))


def save_figure_all_formats(fig: plt.Figure, basepath: str | Path) -> dict[str, Path]:
    """Save in all common publication formats. Returns dict of format → path."""
    base = Path(basepath)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in ("pdf", "svg", "png"):
        p = base.with_suffix(f".{fmt}")
        fig.savefig(p)
        paths[fmt] = p
    return paths
