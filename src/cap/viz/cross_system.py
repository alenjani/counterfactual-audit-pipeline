"""Cross-system comparison plots — radar charts, system×task grids."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cap.viz.theme import CAP_PALETTES


def system_comparison_radar(
    df: pd.DataFrame,
    *,
    system_col: str = "auditor",
    metric_cols: list[str] | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
) -> plt.Figure:
    """Radar chart comparing systems across multiple fairness metrics.

    Higher = better (1 - error_rate). Each axis is a metric (e.g., gender accuracy,
    age MAE, emotion macro-F1, demographic parity, equalized odds).
    """
    if metric_cols is None:
        metric_cols = [c for c in df.columns if c != system_col]
    systems = df[system_col].tolist()
    n_metrics = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    palette = CAP_PALETTES["systems"]
    for i, sys_name in enumerate(systems):
        values = df.loc[df[system_col] == sys_name, metric_cols].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=palette[i % len(palette)], linewidth=2, label=sys_name)
        ax.fill(angles, values, color=palette[i % len(palette)], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_cols, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
    fig.tight_layout()
    return fig


def system_task_grid(
    df: pd.DataFrame,
    *,
    system_col: str = "auditor",
    task_col: str = "task",
    metric_col: str = "accuracy",
    figsize: tuple[float, float] = (7.0, 4.0),
) -> plt.Figure:
    """Heatmap: rows = systems, columns = tasks, color = metric value."""
    import seaborn as sns

    pivot = df.pivot(index=system_col, columns=task_col, values=metric_col)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": metric_col.replace("_", " ").title(), "shrink": 0.8})
    ax.set_xlabel("Task")
    ax.set_ylabel("System")
    fig.tight_layout()
    return fig
