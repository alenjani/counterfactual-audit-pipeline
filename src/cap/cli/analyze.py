"""CLI: run statistical analyses on audit predictions."""
from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from cap.analysis import (
    counterfactual_flip_rate,
    intersectional_error_table,
    mcnemars_paired,
    two_way_anova,
)
from cap.utils import RunManifest, get_logger, load_config

logger = get_logger()


@click.command()
@click.option("--config", "config_path", required=True)
def main(config_path: str) -> None:
    cfg = load_config(config_path)
    analysis_dir = Path(cfg["paths.analysis_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest.create(
        run_id=cfg.get("run_id", "unknown") + "_analysis",
        config_path=config_path,
        config_resolved=cfg.to_dict(),
    )

    audit_dir = Path(cfg["paths.audit_dir"])
    predictions = pd.read_parquet(audit_dir / "predictions.parquet")
    logger.info(f"Loaded {len(predictions)} audit predictions")

    results: dict = {}

    # 1. Counterfactual flip rate per system
    flip_rates = (
        predictions
        .groupby(["auditor", "task"])
        .apply(lambda d: counterfactual_flip_rate(d).iloc[0]["flip_rate"])
        .reset_index(name="flip_rate")
    )
    flip_rates.to_csv(analysis_dir / "flip_rates.csv", index=False)
    results["flip_rates"] = flip_rates.to_dict(orient="records")

    # 2. Intersectional error tables (Gender Shades style) per (system × task)
    # Requires ground-truth column — assumes axis values double as ground truth for now.
    if "axis_gender" in predictions.columns and "axis_skin_tone" in predictions.columns:
        for (auditor, task), sub in predictions.groupby(["auditor", "task"]):
            if task not in {"gender"}:  # extend as needed
                continue
            sub = sub.assign(ground_truth=sub["axis_gender"])
            tbl = intersectional_error_table(
                sub, skin_tone_col="axis_skin_tone", gender_col="axis_gender"
            )
            tbl.to_csv(analysis_dir / f"intersectional_{auditor}_{task}.csv", index=False)

    # 3. Two-way ANOVA on error rate by (skin_tone × gender) per system
    # TODO: wire once ground_truth column is canonicalized.

    # 4. McNemar's paired test (skin tone 1 vs 6)
    # TODO: wire once predictions are properly aligned.

    manifest.finish()
    manifest.write(analysis_dir / "run_manifest.json")
    (analysis_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Analysis complete. Outputs in {analysis_dir}")


if __name__ == "__main__":
    main()
