"""Distribution plots: violin, ridge (joyplot), histograms."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cap.viz.theme import CAP_PALETTES


def confidence_violin(
    df: pd.DataFrame,
    *,
    confidence_col: str = "confidence",
    group_col: str = "skin_tone",
    hue_col: str | None = "gender",
    title: str | None = None,
    figsize: tuple[float, float] = (6.0, 3.5),
) -> plt.Figure:
    """Violin plot of confidence scores by demographic subgroup."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=df, x=group_col, y=confidence_col, hue=hue_col, ax=ax,
        split=hue_col is not None, inner="quart", cut=0, linewidth=1,
        palette=CAP_PALETTES["fitzpatrick"] if group_col == "skin_tone" and hue_col is None else None,
    )
    ax.set_ylabel("Prediction confidence")
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def ridge_skin_tone(
    df: pd.DataFrame,
    *,
    value_col: str,
    skin_tone_col: str = "skin_tone",
    figsize: tuple[float, float] = (5.5, 4.5),
) -> plt.Figure:
    """Ridge (joyplot) of a continuous variable across Fitzpatrick types."""
    sns.set_style("white")
    types = sorted(df[skin_tone_col].unique())
    fig, axes = plt.subplots(len(types), 1, figsize=figsize, sharex=True)
    for ax, t, color in zip(axes, types, CAP_PALETTES["fitzpatrick"]):
        subset = df[df[skin_tone_col] == t][value_col].dropna()
        if len(subset) > 1:
            sns.kdeplot(subset, ax=ax, fill=True, color=color, alpha=0.85, linewidth=0.5)
        ax.set_yticks([])
        ax.set_ylabel(f"Type {t}", rotation=0, ha="right", va="center", fontsize=8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(ax == axes[-1])
    axes[-1].set_xlabel(value_col.replace("_", " ").title())
    fig.subplots_adjust(hspace=-0.4)
    return fig


def score_histogram(
    values: np.ndarray | pd.Series,
    *,
    threshold: float | None = None,
    title: str = "",
    xlabel: str = "Score",
    bins: int = 40,
    figsize: tuple[float, float] = (5.0, 3.0),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, color="#2E5EAA", alpha=0.8, edgecolor="white", linewidth=0.5)
    if threshold is not None:
        ax.axvline(threshold, color="#D62828", linestyle="--", linewidth=1.5, label=f"threshold = {threshold}")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
