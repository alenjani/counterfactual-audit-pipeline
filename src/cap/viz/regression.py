"""Continuous skin-tone regression plot — error rate vs Fitzpatrick scale per system."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cap.viz.theme import CAP_PALETTES


def skin_tone_regression_plot(
    df: pd.DataFrame,
    *,
    skin_tone_col: str = "skin_tone",
    error_col: str = "error_rate",
    system_col: str = "auditor",
    figsize: tuple[float, float] = (6.0, 4.0),
) -> plt.Figure:
    """Scatter + regression line per system, error rate vs continuous Fitzpatrick."""
    fig, ax = plt.subplots(figsize=figsize)
    systems = df[system_col].unique()
    palette = dict(zip(systems, CAP_PALETTES["systems"][:len(systems)]))

    for sys_name in systems:
        sub = df[df[system_col] == sys_name]
        sns.regplot(
            data=sub, x=skin_tone_col, y=error_col, ax=ax,
            label=sys_name, ci=95, scatter_kws={"s": 30, "alpha": 0.7},
            color=palette[sys_name],
        )
    ax.set_xlabel("Fitzpatrick skin type")
    ax.set_ylabel("Error rate")
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.legend(title="System", frameon=True, loc="best")
    fig.tight_layout()
    return fig
