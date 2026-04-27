"""Counterfactual fidelity plots — ArcFace identity similarity, attribute residuals."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def identity_similarity_distribution(
    df: pd.DataFrame,
    *,
    similarity_col: str = "cosine_similarity",
    threshold: float = 0.5,
    group_col: str | None = "counterfactual_axis",
    figsize: tuple[float, float] = (6.0, 3.5),
) -> plt.Figure:
    """ArcFace cosine similarity distribution, by counterfactual axis if provided."""
    fig, ax = plt.subplots(figsize=figsize)
    if group_col and group_col in df.columns:
        sns.violinplot(data=df, x=group_col, y=similarity_col, ax=ax, inner="quart",
                       cut=0, linewidth=1, palette="Set2")
    else:
        sns.histplot(data=df, x=similarity_col, bins=40, ax=ax, color="#2E5EAA",
                     alpha=0.8, edgecolor="white")
    ax.axhline(threshold, color="#D62828", linestyle="--", linewidth=1.5,
               label=f"identity threshold = {threshold}")
    ax.set_ylabel("ArcFace cosine similarity")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig
