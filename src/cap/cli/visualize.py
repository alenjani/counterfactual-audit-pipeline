"""CLI: build all paper-quality figures + interactive dashboard from analysis outputs."""
from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from cap.utils import get_logger, load_config
from cap.viz import (
    apply_paper_style,
    build_interactive_dashboard,
    confidence_violin,
    counterfactual_flip_heatmap,
    identity_similarity_distribution,
    intersectional_error_heatmap,
    save_figure_all_formats,
    skin_tone_regression_plot,
    system_comparison_radar,
    system_task_grid,
)

logger = get_logger()


@click.command()
@click.option("--config", "config_path", required=True)
def main(config_path: str) -> None:
    cfg = load_config(config_path)
    apply_paper_style(dpi=cfg.get("viz.dpi", 300))

    analysis_dir = Path(cfg["paths.analysis_dir"])
    audit_dir = Path(cfg["paths.audit_dir"])
    viz_dir = Path(cfg["paths.viz_dir"])
    figs_dir = viz_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    formats = cfg.get("viz.formats", ["pdf", "png"])

    # Build static figures from CSV outputs (graceful skip if not yet present)
    for csv_path in analysis_dir.glob("intersectional_*.csv"):
        df = pd.read_csv(csv_path)
        fig = intersectional_error_heatmap(df, skin_tone_col="axis_skin_tone",
                                           gender_col="axis_gender", value_col="error_rate",
                                           title=csv_path.stem)
        save_figure_all_formats(fig, figs_dir / csv_path.stem)
        logger.info(f"Saved {csv_path.stem}")

    flip_path = analysis_dir / "flip_rates.csv"
    if flip_path.exists():
        df = pd.read_csv(flip_path)
        # Build a proper flip-transition matrix from raw audit data (placeholder for now).
        logger.info(f"Flip rates loaded; full transition heatmap requires per-pair data")

    # Interactive dashboard
    if cfg.get("viz.build_interactive_dashboard", True):
        predictions = pd.read_parquet(audit_dir / "predictions.parquet")
        out_html = viz_dir / "dashboard.html"
        build_interactive_dashboard(predictions, output_html=out_html)
        logger.info(f"Interactive dashboard: {out_html}")

    logger.info(f"Visualizations written to {viz_dir}")


if __name__ == "__main__":
    main()
