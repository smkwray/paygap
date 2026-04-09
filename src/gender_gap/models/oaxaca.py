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

from gender_gap.utils.weights import confidence_interval, replicate_weight_columns, sdr_summary

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
    valid_m = y_m.notna() & X_m.notna().all(axis=1) & w_m.notna() & w_m.gt(0)
    valid_f = y_f.notna() & X_f.notna().all(axis=1) & w_f.notna() & w_f.gt(0)
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


def oaxaca_unexplained_pct_sdr(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    controls: list[str] | None = None,
    weight_col: str = "person_weight",
    repweight_prefix: str = "PWGTP",
) -> dict[str, float]:
    """Estimate ACS SDR uncertainty for the Oaxaca unexplained share."""
    result = oaxaca_blinder(df, outcome=outcome, controls=controls, weight_col=weight_col)
    rep_cols = replicate_weight_columns(df.columns, prefix=repweight_prefix, main_weight=weight_col)
    if not rep_cols:
        raise ValueError("No ACS replicate-weight columns found")

    rep_estimates = []
    for rep_col in rep_cols:
        try:
            rep_result = oaxaca_blinder(df, outcome=outcome, controls=controls, weight_col=rep_col)
        except Exception as exc:  # pragma: no cover - defensive for occasional degenerate replicates
            logger.warning("Skipping Oaxaca replicate %s: %s", rep_col, exc)
            continue
        rep_estimates.append(rep_result.unexplained_pct)

    summary = sdr_summary(result.unexplained_pct, rep_estimates)
    return {
        "estimate": float(result.unexplained_pct),
        "se": float(summary["se"]),
        "ci95_low": float(summary["ci95_low"]),
        "ci95_high": float(summary["ci95_high"]),
        "n_replicates": len(rep_estimates),
    }


def oaxaca_unexplained_pct_bootstrap(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    controls: list[str] | None = None,
    weight_col: str = "person_weight",
    n_boot: int = 200,
    random_state: int = 0,
) -> dict[str, float]:
    """Estimate bootstrap uncertainty for the Oaxaca unexplained share."""
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")

    result = oaxaca_blinder(df, outcome=outcome, controls=controls, weight_col=weight_col)
    base = df.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    n_obs = len(base)
    rep_estimates = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n_obs, size=n_obs)
        sample = base.iloc[sample_idx].copy()
        rep_result = oaxaca_blinder(sample, outcome=outcome, controls=controls, weight_col=weight_col)
        rep_estimates.append(rep_result.unexplained_pct)

    rep_estimates = np.asarray(rep_estimates, dtype=float)
    se = float(np.nanstd(rep_estimates, ddof=1))
    ci95_low, ci95_high = np.nanpercentile(rep_estimates, [2.5, 97.5])
    return {
        "estimate": float(result.unexplained_pct),
        "se": se,
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "n_replicates": int(n_boot),
    }


def recentered_confidence_interval(
    estimate: float,
    standard_error: float,
    level: float = 0.95,
) -> tuple[float, float]:
    """Center a symmetric interval on a supplied point estimate."""
    return confidence_interval(estimate, standard_error, level=level)
