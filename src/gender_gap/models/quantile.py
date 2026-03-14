"""Quantile regression for distributional gap analysis.

The gender gap varies across the wage distribution. This module
estimates the gap at multiple quantiles to reveal patterns like:
- glass ceiling (larger gap at top)
- sticky floor (larger gap at bottom)
- uniform gap across distribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

logger = logging.getLogger(__name__)


@dataclass
class QuantileResult:
    """Result of quantile regression at a single quantile."""

    quantile: float
    female_coef: float
    female_se: float
    female_pvalue: float
    n_obs: int


def run_quantile_regression(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    controls: list[str] | None = None,
    quantiles: list[float] | None = None,
    weight_col: str = "person_weight",
) -> list[QuantileResult]:
    """Run quantile regression at multiple quantiles.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data.
    outcome : str
        Dependent variable.
    controls : list[str]
        Control variables (plus 'female' is always included).
    quantiles : list[float]
        Quantile levels (0-1). Default: [0.10, 0.25, 0.50, 0.75, 0.90].
    weight_col : str
        Survey weight column (used for weighted bootstrap if available).

    Returns
    -------
    list[QuantileResult]
    """
    if controls is None:
        controls = ["age", "age_sq"]

    if quantiles is None:
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    if outcome not in df.columns and outcome == "log_hourly_wage_real":
        df = df.copy()
        df["log_hourly_wage_real"] = np.log(
            df["hourly_wage_real"].replace(0, np.nan)
        )

    feature_cols = ["female"] + [c for c in controls if c in df.columns]

    # Clean data
    valid = df[outcome].notna()
    for c in feature_cols:
        valid = valid & df[c].notna()
    clean = df[valid].copy()

    if len(clean) < 50:
        raise ValueError(f"Too few observations for quantile regression: {len(clean)}")

    X = sm.add_constant(clean[feature_cols].astype(float))
    y = clean[outcome]

    results = []
    for q in quantiles:
        try:
            model = QuantReg(y, X).fit(q=q)
            if "female" in model.params:
                results.append(QuantileResult(
                    quantile=q,
                    female_coef=model.params["female"],
                    female_se=model.bse["female"],
                    female_pvalue=model.pvalues["female"],
                    n_obs=int(model.nobs),
                ))
                logger.info(
                    "Q%.2f: female=%.4f (SE=%.4f)",
                    q, model.params["female"], model.bse["female"],
                )
        except Exception as e:
            logger.warning("Quantile %.2f failed: %s", q, e)

    return results


def quantile_results_to_dataframe(results: list[QuantileResult]) -> pd.DataFrame:
    """Convert quantile results to a summary DataFrame."""
    return pd.DataFrame([{
        "quantile": r.quantile,
        "female_coef": r.female_coef,
        "female_se": r.female_se,
        "female_pvalue": r.female_pvalue,
        "n_obs": r.n_obs,
    } for r in results])


def diagnose_distributional_pattern(results: list[QuantileResult]) -> str:
    """Diagnose whether the gap follows glass ceiling, sticky floor, or uniform pattern.

    Returns a descriptive string.
    """
    if len(results) < 3:
        return "insufficient_quantiles"

    coefs = {r.quantile: r.female_coef for r in results}

    # Get bottom and top quantile gaps
    sorted_q = sorted(coefs.keys())
    bottom = coefs[sorted_q[0]]
    top = coefs[sorted_q[-1]]
    coefs.get(0.5, coefs[sorted_q[len(sorted_q) // 2]])

    # All coefficients should be negative (women earn less)
    # More negative = larger gap
    if abs(top) > abs(bottom) * 1.25:
        return "glass_ceiling"  # gap widens at top
    elif abs(bottom) > abs(top) * 1.25:
        return "sticky_floor"  # gap widens at bottom
    else:
        return "approximately_uniform"
