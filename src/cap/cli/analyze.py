"""CLI: run statistical analyses on audit predictions.

Produces the confirmatory tests for the paper:

  H1 — two-way ANOVA on error rate (skin_tone × gender) interaction
  H2 — McNemar's paired test (skin tone 1 vs 6) per system
  H3 — ordinal logit on continuous skin tone effect

All p-values are FDR-corrected (Benjamini-Hochberg, alpha from config).

Plus:
  - Coarse counterfactual flip rate (any-flip-per-identity)
  - Pairwise flip rate (fraction of pair-comparisons that disagree — effect size)
  - Intersectional error tables (Gender Shades style) per (auditor × task)
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from cap.analysis import (
    counterfactual_flip_rate,
    fdr_correct,
    intersectional_error_table,
    mcnemars_paired,
    ordinal_logit_skin_tone,
    pairwise_flip_rate,
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
    logger.info(f"Loaded {len(predictions)} audit predictions; columns={list(predictions.columns)}")

    # ---- Label normalization (per Review 001 §4 follow-up) -------------------
    # Auditors return service-native labels (DeepFace: "Man"/"Woman", AWS
    # Rekognition: "Male"/"Female", Azure: "male"/"female", Face++: "M"/"F",
    # Google Vision: "Likely"/"Possibly"/"Unlikely" with gender, ...). Without
    # normalization, a string compare to axis_gender ("male"/"female") marks
    # every prediction as wrong → ANOVA has zero variance → empty H1/H3.
    # Normalize predictions in-place to the canonical {male, female} alphabet
    # used by axis_gender. Unknown values are left as-is (will still register
    # as errors but at least correctly rather than systemically).
    _GENDER_ALIASES = {
        # DeepFace
        "man": "male", "woman": "female",
        # AWS Rekognition
        "male": "male", "female": "female",
        # Face++
        "m": "male", "f": "female",
        # Misc
        "boy": "male", "girl": "female",
    }
    if "task" in predictions.columns and "prediction" in predictions.columns:
        gender_mask = predictions["task"] == "gender"
        n_to_norm = int(gender_mask.sum())
        if n_to_norm > 0:
            normalized = (
                predictions.loc[gender_mask, "prediction"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(lambda v: _GENDER_ALIASES.get(v, v))
            )
            predictions.loc[gender_mask, "prediction"] = normalized
            uniq = sorted(predictions.loc[gender_mask, "prediction"].dropna().unique().tolist())
            logger.info(
                f"Normalized {n_to_norm} gender-task predictions; "
                f"resulting alphabet: {uniq}"
            )

    has_skin_tone = "axis_skin_tone" in predictions.columns
    has_gender = "axis_gender" in predictions.columns
    if not (has_skin_tone and has_gender):
        logger.warning(
            f"axis_skin_tone={has_skin_tone}, axis_gender={has_gender} — "
            "intersectional and ANOVA analyses require both. Skipping those."
        )

    fdr_alpha = float(cfg.get("analysis.fdr_alpha", 0.05))

    results: dict = {"fdr_alpha": fdr_alpha}

    # ---- 0. Treat axis_gender as ground truth for the gender task -------------
    # When the auditor task is "gender", the prompt-fixed axis_gender IS the
    # ground truth (we asked the model to make this person look male/female).
    # For "age", axis values are ints (30) so we don't compute error/accuracy
    # in the same way — flip rate is the relevant metric.
    df = predictions.copy()
    df["error"] = np.where(
        (df["task"] == "gender") & has_gender,
        (df["prediction"].astype(str).str.lower() != df["axis_gender"].astype(str).str.lower()).astype(int),
        np.nan,
    )

    # ---- 1. Coarse and pairwise flip rates per (auditor, task) ----------------
    flip_rows = []
    pairwise_rows = []
    for (auditor, task), sub in predictions.groupby(["auditor", "task"]):
        coarse = counterfactual_flip_rate(sub).iloc[0]
        flip_rows.append({
            "auditor": auditor, "task": task,
            "coarse_flip_rate": float(coarse["flip_rate"]),
            "n_identities": int(coarse["n_identities"]),
        })
        pw = pairwise_flip_rate(sub)
        if len(pw):
            pairwise_rows.append({
                "auditor": auditor, "task": task,
                "mean_pairwise_flip_rate": float(pw["pairwise_flip_rate"].mean()),
                "median_pairwise_flip_rate": float(pw["pairwise_flip_rate"].median()),
                "std_pairwise_flip_rate": float(pw["pairwise_flip_rate"].std()),
                "n_identities": int(len(pw)),
            })
            pw["auditor"] = auditor
            pw["task"] = task
            (analysis_dir / f"pairwise_per_identity_{auditor}_{task}.csv").write_text(
                pw.to_csv(index=False)
            )

    flip_df = pd.DataFrame(flip_rows)
    flip_df.to_csv(analysis_dir / "flip_rates.csv", index=False)
    results["flip_rates"] = flip_df.to_dict(orient="records")

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(analysis_dir / "pairwise_flip_rates.csv", index=False)
    results["pairwise_flip_rates"] = pairwise_df.to_dict(orient="records")
    logger.info(f"Flip rates: {len(flip_df)} (auditor, task) combos. Pairwise: {len(pairwise_df)}.")

    # ---- 2. Intersectional error tables per (auditor × task=gender) -----------
    if has_skin_tone and has_gender:
        for (auditor, task), sub in predictions.groupby(["auditor", "task"]):
            if task != "gender":
                continue
            sub = sub.assign(
                ground_truth=sub["axis_gender"].astype(str).str.lower(),
                prediction_norm=sub["prediction"].astype(str).str.lower(),
            )
            tbl = intersectional_error_table(
                sub,
                skin_tone_col="axis_skin_tone",
                gender_col="axis_gender",
                truth_col="ground_truth",
                prediction_col="prediction_norm",
            )
            tbl.to_csv(analysis_dir / f"intersectional_{auditor}_{task}.csv", index=False)
            logger.info(f"Wrote intersectional_{auditor}_{task}.csv ({len(tbl)} rows)")

    # ---- 3. H1: Two-way ANOVA per (auditor, task=gender) ---------------------
    anova_rows = []
    if has_skin_tone and has_gender:
        for (auditor, task), sub in predictions.groupby(["auditor", "task"]):
            if task != "gender":
                continue
            sub2 = df[(df["auditor"] == auditor) & (df["task"] == task)].copy()
            sub2 = sub2.dropna(subset=["error"])
            if sub2["error"].nunique() < 2:
                logger.info(f"H1 skipped for {auditor}/{task}: error has no variance "
                            f"({sub2['error'].sum()}/{len(sub2)} errors)")
                continue
            try:
                aov = two_way_anova(sub2, dv="error", factor_a="axis_skin_tone", factor_b="axis_gender")
                interaction = aov[aov["Source"].str.contains("axis_skin_tone \\* axis_gender", regex=True)]
                p_int = float(interaction.iloc[0]["p-unc"]) if len(interaction) else float("nan")
                anova_rows.append({
                    "auditor": auditor, "task": task,
                    "p_skin_tone": float(aov[aov["Source"] == "axis_skin_tone"].iloc[0]["p-unc"]),
                    "p_gender": float(aov[aov["Source"] == "axis_gender"].iloc[0]["p-unc"]),
                    "p_interaction": p_int,
                    "n": len(sub2),
                })
            except Exception as e:
                logger.warning(f"ANOVA failed for {auditor}/{task}: {e}")

    anova_df = pd.DataFrame(anova_rows)
    if len(anova_df):
        # FDR-correct the family of interaction p-values across systems
        rejected, corrected = fdr_correct(anova_df["p_interaction"].fillna(1.0).tolist(), alpha=fdr_alpha)
        anova_df["p_interaction_fdr"] = corrected
        anova_df["H1_supported"] = rejected
    anova_df.to_csv(analysis_dir / "h1_anova.csv", index=False)
    results["h1_anova"] = anova_df.to_dict(orient="records")
    logger.info(f"H1 (ANOVA): {len(anova_df)} systems tested.")

    # ---- 4. H2: McNemar's paired flip test per (auditor, task=gender) --------
    mcn_rows = []
    if has_skin_tone:
        for (auditor, task), sub in predictions.groupby(["auditor", "task"]):
            if task != "gender":
                continue
            sub2 = df[(df["auditor"] == auditor) & (df["task"] == task)].copy()
            sub2 = sub2.dropna(subset=["error"])
            try:
                res = mcnemars_paired(
                    sub2,
                    pair_col="seed_identity_id",
                    condition_col="axis_skin_tone",
                    outcome_col="error",
                    cond_a_value=1,
                    cond_b_value=6,
                )
                mcn_rows.append({"auditor": auditor, "task": task, **res})
            except Exception as e:
                logger.warning(f"McNemar's failed for {auditor}/{task}: {e}")

    mcn_df = pd.DataFrame(mcn_rows)
    if len(mcn_df):
        rejected, corrected = fdr_correct(mcn_df["pvalue"].fillna(1.0).tolist(), alpha=fdr_alpha)
        mcn_df["pvalue_fdr"] = corrected
        mcn_df["H2_supported"] = rejected
    mcn_df.to_csv(analysis_dir / "h2_mcnemars.csv", index=False)
    results["h2_mcnemars"] = mcn_df.to_dict(orient="records")
    logger.info(f"H2 (McNemar's): {len(mcn_df)} systems tested.")

    # ---- 5. H3: Ordinal/continuous skin tone effect (logit) ------------------
    logit_rows = []
    if has_skin_tone:
        for (auditor, task), sub in predictions.groupby(["auditor", "task"]):
            if task != "gender":
                continue
            sub2 = df[(df["auditor"] == auditor) & (df["task"] == task)].copy()
            sub2 = sub2.dropna(subset=["error"]).assign(
                axis_skin_tone=lambda d: d["axis_skin_tone"].astype(int),
                gender_num=lambda d: (d["axis_gender"].astype(str).str.lower() == "female").astype(int),
            )
            if sub2["error"].nunique() < 2:
                continue
            try:
                model = ordinal_logit_skin_tone(
                    sub2,
                    skin_tone_col="axis_skin_tone",
                    error_col="error",
                    covariates=["gender_num"],
                )
                logit_rows.append({
                    "auditor": auditor, "task": task,
                    "skin_tone_coef": float(model.params["axis_skin_tone"]),
                    "skin_tone_p": float(model.pvalues["axis_skin_tone"]),
                    "skin_tone_odds_ratio": float(np.exp(model.params["axis_skin_tone"])),
                    "n": int(model.nobs),
                })
            except Exception as e:
                logger.warning(f"Logit failed for {auditor}/{task}: {e}")

    logit_df = pd.DataFrame(logit_rows)
    if len(logit_df):
        rejected, corrected = fdr_correct(logit_df["skin_tone_p"].fillna(1.0).tolist(), alpha=fdr_alpha)
        logit_df["skin_tone_p_fdr"] = corrected
        logit_df["H3_supported"] = rejected
    logit_df.to_csv(analysis_dir / "h3_ordinal_logit.csv", index=False)
    results["h3_ordinal_logit"] = logit_df.to_dict(orient="records")
    logger.info(f"H3 (ordinal logit): {len(logit_df)} systems tested.")

    # ---- 6. Identity preservation (joined from validation, if present) -------
    validation_dir = Path(cfg.get("paths.validation_dir", str(Path(cfg["paths.output_dir"]) / "validation")))
    id_path = validation_dir / "identity_scores.parquet"
    if id_path.exists():
        idd = pd.read_parquet(id_path)
        results["identity"] = {
            "mean_cosine_sim": float(idd["cosine_similarity"].mean()),
            "median_cosine_sim": float(idd["cosine_similarity"].median()),
            "preserved_fraction": float(idd["is_preserved"].mean()),
            "n": int(len(idd)),
        }
        logger.info(f"Identity: mean_sim={results['identity']['mean_cosine_sim']:.3f}, "
                    f"preserved={results['identity']['preserved_fraction']:.1%}")
    else:
        logger.info(f"No identity scores found at {id_path} — skipping identity summary.")

    manifest.finish()
    manifest.write(analysis_dir / "run_manifest.json")
    (analysis_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Analysis complete. Outputs in {analysis_dir}")


if __name__ == "__main__":
    main()
