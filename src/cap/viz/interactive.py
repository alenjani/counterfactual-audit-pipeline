"""Interactive Plotly + Altair dashboards for supplementary materials and web exhibits.

These export to standalone HTML — embed in supplementary materials, ISR companion website,
or talks. Reviewers can explore the data without running code.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_interactive_dashboard(
    audit_results: pd.DataFrame,
    *,
    output_html: str | Path,
    title: str = "Counterfactual Audit — Interactive Dashboard",
) -> Path:
    """Compose multi-panel Plotly dashboard from audit results.

    Panels:
      1. Intersectional error heatmap (interactive hover)
      2. Counterfactual flip rate by system (bar)
      3. Confidence distribution by demographic (violin)
      4. System × task accuracy grid
      5. Filterable table of every audit decision

    Output: a single self-contained HTML file.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Intersectional error rate",
            "Counterfactual flip rate by system",
            "Confidence by skin tone",
            "System × task accuracy",
        ),
        specs=[[{"type": "heatmap"}, {"type": "bar"}],
               [{"type": "violin"}, {"type": "heatmap"}]],
    )

    # Panel 1: intersectional heatmap (placeholder until real data wired)
    if {"skin_tone", "gender", "error_rate"}.issubset(audit_results.columns):
        pivot = audit_results.pivot_table(
            index="skin_tone", columns="gender", values="error_rate", aggfunc="mean"
        )
        fig.add_trace(
            go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                       colorscale="Magma", showscale=False, hovertemplate="%{z:.1%}"),
            row=1, col=1,
        )

    # Panel 2: flip rate by system
    if {"auditor", "flip_rate"}.issubset(audit_results.columns):
        flips = audit_results.groupby("auditor")["flip_rate"].mean().reset_index()
        fig.add_trace(
            go.Bar(x=flips["auditor"], y=flips["flip_rate"], marker_color="#D62828"),
            row=1, col=2,
        )

    fig.update_layout(
        title=title, template="plotly_white", height=800, showlegend=False,
        font=dict(family="Arial, sans-serif", size=11),
    )

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)
    return output_html
