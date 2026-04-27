"""Statistical tests for the audit framework."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def two_way_anova(
    df: pd.DataFrame,
    dv: str,
    factor_a: str,
    factor_b: str,
) -> pd.DataFrame:
    """Two-way ANOVA via pingouin for clean output."""
    import pingouin as pg

    return pg.anova(data=df, dv=dv, between=[factor_a, factor_b], detailed=True)


def mcnemars_paired(
    df: pd.DataFrame,
    pair_col: str,
    condition_col: str,
    outcome_col: str,
    cond_a_value,
    cond_b_value,
) -> dict[str, float]:
    """McNemar's test on paired counterfactual outcomes.

    For each `pair_col` (e.g., seed_identity_id), compare the binary outcome
    between two conditions (e.g., skin_tone=1 vs skin_tone=6).
    """
    pivot = df.pivot_table(index=pair_col, columns=condition_col, values=outcome_col, aggfunc="first")
    pivot = pivot.dropna(subset=[cond_a_value, cond_b_value])
    a = pivot[cond_a_value].astype(int).values
    b = pivot[cond_b_value].astype(int).values
    table = np.array([
        [((a == 1) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum()],
        [((a == 0) & (b == 1)).sum(), ((a == 0) & (b == 0)).sum()],
    ])
    res = mcnemar(table, exact=False, correction=True)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue), "n_pairs": len(pivot)}


def ordinal_logit_skin_tone(
    df: pd.DataFrame,
    skin_tone_col: str = "skin_tone",
    error_col: str = "error",
    covariates: list[str] | None = None,
) -> "any":
    """Ordinal logistic regression: error ~ skin_tone (continuous Fitzpatrick) + covariates."""
    import statsmodels.formula.api as smf

    formula = f"{error_col} ~ {skin_tone_col}"
    if covariates:
        formula += " + " + " + ".join(covariates)
    model = smf.logit(formula, data=df).fit(disp=False)
    return model


def repeated_measures_anova(
    df: pd.DataFrame, dv: str, within: list[str], subject: str
) -> pd.DataFrame:
    """Repeated measures ANOVA across systems / tasks within identity."""
    import pingouin as pg

    return pg.rm_anova(data=df, dv=dv, within=within, subject=subject, detailed=True)


def fdr_correct(pvalues: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction. Returns (rejected, corrected_pvalues)."""
    rejected, corrected, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    return rejected, corrected
