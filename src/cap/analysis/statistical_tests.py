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
    """Two-way analysis with main + interaction effects.

    Auto-routes to the right model:
    - **Continuous DV** → pingouin's classical two-way ANOVA.
    - **Binary DV** (only {0, 1} values) → logistic regression with the same
      `factor_a * factor_b` design, returned as a same-shape `Source / p-unc`
      DataFrame so callers don't change. ANOVA on a binary DV is misspecified
      and tends to fail with `LinAlgError: Singular matrix` whenever any cell
      has zero residual variance (which happens routinely with small per-cell
      samples + low error rate).

    Output schema:
        DataFrame with columns ['Source', 'p-unc', 'F', 'np2'] (np2 left as
        NaN for the binary path, which doesn't have an η²-equivalent).
        `Source` values include factor_a, factor_b, and the interaction
        `f"{factor_a} * {factor_b}"` — caller code in analyze.py searches
        these names so they must be preserved.
    """
    is_binary = (
        df[dv].dropna().isin([0, 1, 0.0, 1.0]).all()
        and df[dv].nunique(dropna=True) <= 2
    )
    if is_binary:
        return _two_way_logit(df, dv, factor_a, factor_b)
    import pingouin as pg
    return pg.anova(data=df, dv=dv, between=[factor_a, factor_b], detailed=True)


def _two_way_logit(
    df: pd.DataFrame, dv: str, factor_a: str, factor_b: str
) -> pd.DataFrame:
    """Two-way logistic regression with interaction. Returns ANOVA-shaped DF.

    Wald test p-values for the joint contribution of each factor (and the
    interaction) — comparable to ANOVA p-values for screening purposes.
    """
    import statsmodels.formula.api as smf

    # Sanitize DV + factor names for patsy: only alnum + underscore
    sub = df[[dv, factor_a, factor_b]].dropna().copy()
    sub[factor_a] = sub[factor_a].astype("category")
    sub[factor_b] = sub[factor_b].astype("category")
    sub[dv] = sub[dv].astype(int)

    formula = f"{dv} ~ C({factor_a}) * C({factor_b})"
    try:
        result = smf.logit(formula, data=sub).fit(disp=False, maxiter=200)
    except Exception as e:
        # Fall through to a "test couldn't run" row so callers see something
        return pd.DataFrame([
            {"Source": factor_a, "p-unc": float("nan"), "F": float("nan"), "np2": float("nan")},
            {"Source": factor_b, "p-unc": float("nan"), "F": float("nan"), "np2": float("nan")},
            {"Source": f"{factor_a} * {factor_b}", "p-unc": float("nan"),
             "F": float("nan"), "np2": float("nan"), "_error": str(e)[:200]},
        ])

    # Wald test for joint significance of each factor's coefficient set.
    rows = []
    for source, terms in [
        (factor_a, [t for t in result.params.index if t.startswith(f"C({factor_a})") and ":" not in t]),
        (factor_b, [t for t in result.params.index if t.startswith(f"C({factor_b})") and ":" not in t]),
        (f"{factor_a} * {factor_b}", [t for t in result.params.index if ":" in t]),
    ]:
        if not terms:
            rows.append({"Source": source, "p-unc": float("nan"), "F": float("nan"), "np2": float("nan")})
            continue
        try:
            wald = result.wald_test(terms, scalar=True)
            rows.append({
                "Source": source,
                "p-unc": float(wald.pvalue),
                "F": float(wald.statistic),  # actually a chi-sq statistic; named F for schema parity
                "np2": float("nan"),
            })
        except Exception as e:
            rows.append({
                "Source": source, "p-unc": float("nan"),
                "F": float("nan"), "np2": float("nan"), "_error": str(e)[:200],
            })
    return pd.DataFrame(rows)


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
