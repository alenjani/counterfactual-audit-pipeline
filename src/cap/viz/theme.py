"""Publication-quality matplotlib + seaborn theme.

Goals:
  - Vector-clean output (PDF/SVG with editable text)
  - Perceptually uniform colormaps (cividis, viridis, colorcet)
  - Type that survives Times New Roman / journal house style
  - Generous whitespace, no chartjunk
  - Accessibility: deuteranopia-safe palettes
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Colorblind-safe palette tuned for fairness viz (warm = bias, cool = fair)
CAP_COLORS = {
    "primary": "#2E5EAA",      # deep blue — neutral / "fair" baseline
    "accent": "#D62828",        # red — bias / disparity
    "neutral": "#6C757D",       # slate
    "highlight": "#F77F00",     # amber — attention without alarm
    "subtle": "#A8DADC",        # light teal — supporting
    "background": "#F8F9FA",    # near-white panel bg
    "text": "#212529",          # near-black text
}

CAP_PALETTES = {
    # Sequential — for skin-tone gradients (Fitzpatrick I → VI), perceptually uniform
    "fitzpatrick": ["#FFE4C4", "#F4C28C", "#D89F73", "#A87858", "#704F3C", "#3B2820"],
    # Categorical — for cross-system comparison (7 systems), colorblind-safe
    "systems": ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#FBAFE4", "#949494"],
    # Diverging — for flip rates (centered on 0)
    "diverging": "RdBu_r",
    # Sequential — for error magnitudes
    "magma": "magma",
}


def apply_paper_style(font_family: str = "serif", font_size: int = 10, dpi: int = 300) -> None:
    """Apply CAP paper style globally. Call once at the start of viz scripts."""
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,
        "pdf.fonttype": 42,           # editable text in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",       # text stays as text in SVG
        "font.family": font_family,
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.titleweight": "bold",
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.5,
        "axes.prop_cycle": plt.cycler("color", CAP_PALETTES["systems"]),
    })
