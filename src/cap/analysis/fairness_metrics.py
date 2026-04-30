"""Fairness metrics — counterfactual flip rate, intersectional error tables."""
from __future__ import annotations

import pandas as pd


def counterfactual_flip_rate(
    df: pd.DataFrame,
    identity_col: str = "seed_identity_id",
    prediction_col: str = "prediction",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Fraction of seed identities where ANY counterfactual produced a different prediction.

    A "flip" = predictions differ across counterfactuals of the same identity.
    Flip rate = #identities with ≥1 flip / total identities.
    Optionally compute per-group within `group_cols`.

    NOTE: this is a coarse 0/1 indicator. For effect-size measurement use
    `pairwise_flip_rate` (fraction of pair comparisons that disagree).
    """
    if group_cols is None:
        flips = df.groupby(identity_col)[prediction_col].nunique() > 1
        rate = flips.mean()
        return pd.DataFrame({"flip_rate": [rate], "n_identities": [len(flips)]})
    flips = df.groupby([identity_col, *group_cols])[prediction_col].nunique() > 1
    rate = flips.groupby(group_cols).mean().reset_index(name="flip_rate")
    return rate


def pairwise_flip_rate(
    df: pd.DataFrame,
    identity_col: str = "seed_identity_id",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Per-identity fraction of pair-comparisons whose predictions disagree.

    For each seed identity with N counterfactuals, compute (number of disagreeing
    unordered pairs) / (N choose 2). Returns one row per identity with the rate.
    Aggregating across identities (mean) gives the overall pairwise flip rate —
    the effect size the paper measures (per H2 in the README).
    """
    rows = []
    for ident, group in df.groupby(identity_col):
        preds = group[prediction_col].tolist()
        n = len(preds)
        if n < 2:
            continue
        n_pairs = n * (n - 1) // 2
        n_disagree = 0
        for i in range(n):
            for j in range(i + 1, n):
                if preds[i] != preds[j]:
                    n_disagree += 1
        rows.append({
            identity_col: ident,
            "n_counterfactuals": n,
            "n_pairs": n_pairs,
            "n_disagree": n_disagree,
            "pairwise_flip_rate": n_disagree / n_pairs,
        })
    return pd.DataFrame(rows)


def subgroup_error_rates(
    df: pd.DataFrame,
    truth_col: str = "ground_truth",
    prediction_col: str = "prediction",
    subgroup_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Error rate per subgroup, in the Gender Shades style."""
    df = df.copy()
    df["correct"] = df[truth_col] == df[prediction_col]
    if subgroup_cols is None:
        return pd.DataFrame({"error_rate": [1 - df["correct"].mean()], "n": [len(df)]})
    grouped = df.groupby(subgroup_cols)["correct"].agg(["count", "mean"]).reset_index()
    grouped["error_rate"] = 1 - grouped["mean"]
    grouped = grouped.rename(columns={"count": "n"}).drop(columns="mean")
    return grouped


def intersectional_error_table(
    df: pd.DataFrame,
    skin_tone_col: str = "skin_tone",
    gender_col: str = "gender",
    truth_col: str = "ground_truth",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Gender Shades-style table: error rates by skin tone × gender."""
    return subgroup_error_rates(df, truth_col, prediction_col, [skin_tone_col, gender_col])
