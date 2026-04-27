"""Intersectional error and counterfactual flip heatmaps.

The KEY visual for the paper — Gender Shades style 6×2 (skin tone × gender) heatmap,
extended to flip-rate visualizations that show the counterfactual contribution.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cap.viz.theme import CAP_PALETTES


def intersectional_error_heatmap(
    df: pd.DataFrame,
    *,
    skin_tone_col: str = "skin_tone",
    gender_col: str = "gender",
    value_col: str = "error_rate",
    title: str | None = None,
    annotate: bool = True,
    vmin: float = 0.0,
    vmax: float = 0.5,
    cmap: str = "magma",
    figsize: tuple[float, float] = (4.0, 3.5),
) -> plt.Figure:
    """Gender Shades-style heatmap of error rate across (skin_tone × gender)."""
    pivot = df.pivot(index=skin_tone_col, columns=gender_col, values=value_col)
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot, ax=ax, annot=annotate, fmt=".1%", cmap=cmap, vmin=vmin, vmax=vmax,
        cbar_kws={"label": value_col.replace("_", " ").title(), "shrink": 0.8},
        linewidths=0.5, linecolor="white", square=False,
    )
    ax.set_xlabel("Gender presentation")
    ax.set_ylabel("Fitzpatrick skin type")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def counterfactual_flip_heatmap(
    flip_matrix: pd.DataFrame,
    *,
    title: str = "Counterfactual flip rate",
    cmap: str = "RdBu_r",
    center: float = 0.0,
    figsize: tuple[float, float] = (5.5, 4.5),
) -> plt.Figure:
    """Heatmap of flip rate across demographic transitions (e.g., from→to skin tone).

    `flip_matrix`: square DataFrame indexed by source demographic, columns = target demographic;
    values = % of identities whose prediction flipped on that transition.
    """
    fig, ax = plt.subplots(figsize=figsize)
    vmax = float(np.nanmax(np.abs(flip_matrix.values))) if flip_matrix.size else 1.0
    sns.heatmap(
        flip_matrix, ax=ax, annot=True, fmt=".1%", cmap=cmap, center=center,
        vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Flip rate", "shrink": 0.8},
    )
    ax.set_xlabel("Counterfactual demographic")
    ax.set_ylabel("Seed demographic")
    ax.set_title(title)
    fig.tight_layout()
    return fig
