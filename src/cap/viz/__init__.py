"""Publication-quality visualization for fairness audits.

Two output families:
  - Static (matplotlib + seaborn): PDF/SVG/PNG for the paper
  - Interactive (Plotly + Altair): HTML for supplementary materials, talks, web exhibits

Every plot function returns a Figure object so callers can compose, save, or further customize.
"""
from cap.viz.theme import apply_paper_style, CAP_COLORS, CAP_PALETTES
from cap.viz.intersectional_heatmap import intersectional_error_heatmap, counterfactual_flip_heatmap
from cap.viz.distributions import confidence_violin, ridge_skin_tone, score_histogram
from cap.viz.regression import skin_tone_regression_plot
from cap.viz.fidelity import identity_similarity_distribution
from cap.viz.cross_system import system_comparison_radar, system_task_grid
from cap.viz.interactive import build_interactive_dashboard
from cap.viz.export import save_figure, save_figure_all_formats

__all__ = [
    "apply_paper_style",
    "CAP_COLORS",
    "CAP_PALETTES",
    "intersectional_error_heatmap",
    "counterfactual_flip_heatmap",
    "confidence_violin",
    "ridge_skin_tone",
    "score_histogram",
    "skin_tone_regression_plot",
    "identity_similarity_distribution",
    "system_comparison_radar",
    "system_task_grid",
    "build_interactive_dashboard",
    "save_figure",
    "save_figure_all_formats",
]
