"""Oaxaca-Blinder decomposition.

Decomposes the male-female wage gap into:
- Explained (endowments): differences in characteristics
- Unexplained (coefficients): differences in returns to characteristics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class OaxacaResult:
    """Result of an Oaxaca-Blinder decomposition."""

    total_gap: float
    explained: float
    unexplained: float
    explained_pct: float
    unexplained_pct: float
    n_male: int
    n_female: int
    contributions: pd.DataFrame  # variable-level explained contributions


def oaxaca_blinder(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    controls: list[str] | None = None,
    weight_col: str = "person_weight",
) -> OaxacaResult:
    """Run a two-fold Oaxaca-Blinder decomposition.

    Uses the pooled (Neumark) reference structure by default.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data with outcome, female indicator, and controls.
    outcome : str
        Dependent variable (typically log_hourly_wage_real).
    controls : list[str]
        List of control variable names. Categorical vars should be
        pre-dummified before calling.
    weight_col : str
        Survey weight column.

    Returns
    -------
    OaxacaResult
    """
    if controls is None:
        controls = [
            "age", "age_sq",
            "usual_hours_week", "work_from_home",
            "commute_minutes_one_way",
            "number_children", "children_under_5",
        ]

    # Prepare log wage if needed
    if outcome not in df.columns and outcome == "log_hourly_wage_real":
        df = df.copy()
        df["log_hourly_wage_real"] = np.log(
            df["hourly_wage_real"].replace(0, np.nan)
        )

    # Split by sex
    male = df[df["female"] == 0].copy()
    female = df[df["female"] == 1].copy()

    # Build X matrices with available controls only
    available = [c for c in controls if c in df.columns]
    if not available:
        raise ValueError("No control variables found in DataFrame")

    X_m = sm.add_constant(male[available].astype(float))
    X_f = sm.add_constant(female[available].astype(float))
    y_m = male[outcome]
    y_f = female[outcome]
    w_m = male[weight_col]
    w_f = female[weight_col]

    # Drop rows with NaN
    valid_m = y_m.notna() & X_m.notna().all(axis=1) & w_m.notna()
    valid_f = y_f.notna() & X_f.notna().all(axis=1) & w_f.notna()
    X_m, y_m, w_m = X_m[valid_m], y_m[valid_m], w_m[valid_m]
    X_f, y_f, w_f = X_f[valid_f], y_f[valid_f], w_f[valid_f]

    # Fit weighted models
    sm.WLS(y_m, X_m, weights=w_m).fit()
    sm.WLS(y_f, X_f, weights=w_f).fit()

    # Pooled model for Neumark reference
    X_all = pd.concat([X_m, X_f])
    y_all = pd.concat([y_m, y_f])
    w_all = pd.concat([w_m, w_f])
    model_p = sm.WLS(y_all, X_all, weights=w_all).fit()

    # Weighted means of characteristics
    mean_m = np.average(X_m.values, weights=w_m.values, axis=0)
    mean_f = np.average(X_f.values, weights=w_f.values, axis=0)

    # Decomposition
    beta_p = model_p.params.values
    diff_means = mean_m - mean_f

    total_gap = np.average(y_m, weights=w_m) - np.average(y_f, weights=w_f)
    explained = np.dot(diff_means, beta_p)
    unexplained = total_gap - explained

    explained_pct = (explained / total_gap * 100) if total_gap != 0 else 0.0
    unexplained_pct = (unexplained / total_gap * 100) if total_gap != 0 else 0.0

    # Variable-level contributions
    var_names = list(X_m.columns)
    contributions = pd.DataFrame({
        "variable": var_names,
        "mean_male": mean_m,
        "mean_female": mean_f,
        "diff_means": diff_means,
        "pooled_coef": beta_p,
        "contribution": diff_means * beta_p,
    })
    contributions["contribution_pct"] = (
        contributions["contribution"] / total_gap * 100
    ) if total_gap != 0 else 0.0

    return OaxacaResult(
        total_gap=total_gap,
        explained=explained,
        unexplained=unexplained,
        explained_pct=explained_pct,
        unexplained_pct=unexplained_pct,
        n_male=len(X_m),
        n_female=len(X_f),
        contributions=contributions,
    )


def oaxaca_summary_table(result: OaxacaResult) -> pd.DataFrame:
    """Create a summary table from an Oaxaca-Blinder result."""
    return pd.DataFrame([
        {"component": "Total gap", "value": result.total_gap,
         "pct": 100.0},
        {"component": "Explained (endowments)", "value": result.explained,
         "pct": result.explained_pct},
        {"component": "Unexplained (coefficients)", "value": result.unexplained,
         "pct": result.unexplained_pct},
    ])
