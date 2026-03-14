"""Heterogeneous residual-gap models.

Estimates where the gender residual gap is larger or smaller:
by parenthood, education, occupation, industry, commute burden,
metro status, age group, and local labor-market conditions.

Uses grouped DML or subgroup-specific models rather than
causal forests (to avoid heavy dependencies for v1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class HeterogeneityResult:
    """Result of heterogeneous gap estimation across subgroups."""

    dimension: str
    subgroup_gaps: pd.DataFrame  # group, gap, se, ci_lower, ci_upper, n


def estimate_heterogeneous_gaps(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    group_col: str = "education_level",
    controls: list[str] | None = None,
    weight_col: str = "person_weight",
) -> HeterogeneityResult:
    """Estimate the gender gap within each subgroup via WLS.

    Runs a separate regression within each subgroup level,
    extracting the female coefficient each time.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data.
    outcome : str
        Dependent variable.
    group_col : str
        Column defining subgroups.
    controls : list[str]
        Control variables (beyond 'female').
    weight_col : str
        Survey weight.
    """
    if controls is None:
        controls = ["age", "age_sq"]

    if outcome not in df.columns and outcome == "log_hourly_wage_real":
        df = df.copy()
        df["log_hourly_wage_real"] = np.log(
            df["hourly_wage_real"].replace(0, np.nan)
        )

    results = []
    for group_val, gdf in df.groupby(group_col, observed=True):
        if len(gdf) < 30:
            continue

        available = [c for c in controls if c in gdf.columns]
        feature_cols = ["female"] + available

        valid = gdf[outcome].notna() & gdf[weight_col].notna()
        for c in feature_cols:
            valid = valid & gdf[c].notna()
        clean = gdf[valid]

        if len(clean) < 20:
            continue

        try:
            X = sm.add_constant(clean[feature_cols].astype(float))
            y = clean[outcome]
            w = clean[weight_col]
            model = sm.WLS(y, X, weights=w).fit()

            if "female" in model.params:
                gap = model.params["female"]
                se = model.bse["female"]
                results.append({
                    "group": group_val,
                    "gap": gap,
                    "se": se,
                    "ci_lower": gap - 1.96 * se,
                    "ci_upper": gap + 1.96 * se,
                    "n": len(clean),
                })
        except Exception as e:
            logger.warning("Heterogeneity %s=%s failed: %s", group_col, group_val, e)

    return HeterogeneityResult(
        dimension=group_col,
        subgroup_gaps=pd.DataFrame(results),
    )


def run_full_heterogeneity(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    dimensions: list[str] | None = None,
    controls: list[str] | None = None,
    weight_col: str = "person_weight",
) -> dict[str, HeterogeneityResult]:
    """Run heterogeneous gap analysis across multiple dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data.
    outcome : str
        Dependent variable.
    dimensions : list[str]
        Columns to stratify by. If None, uses defaults.
    controls : list[str]
        Control variables.
    weight_col : str
        Survey weight.

    Returns
    -------
    dict mapping dimension name -> HeterogeneityResult
    """
    if dimensions is None:
        dimensions = _default_dimensions(df)

    if controls is None:
        controls = ["age", "age_sq"]

    results = {}
    for dim in dimensions:
        if dim not in df.columns:
            logger.warning("Dimension %s not in data, skipping", dim)
            continue
        logger.info("Estimating heterogeneous gap by %s", dim)
        results[dim] = estimate_heterogeneous_gaps(
            df, outcome, dim, controls, weight_col
        )

    return results


def interaction_model(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    interact_col: str = "education_level",
    controls: list[str] | None = None,
    weight_col: str = "person_weight",
) -> pd.DataFrame:
    """Estimate female x subgroup interactions in a single model.

    More efficient than separate regressions and provides
    direct tests of gap differences across groups.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data.
    outcome : str
        Dependent variable.
    interact_col : str
        Column to interact with female.
    controls : list[str]
        Additional controls.
    weight_col : str
        Survey weight.

    Returns
    -------
    pd.DataFrame
        Interaction coefficients with standard errors.
    """
    if controls is None:
        controls = ["age", "age_sq"]

    if outcome not in df.columns and outcome == "log_hourly_wage_real":
        df = df.copy()
        df["log_hourly_wage_real"] = np.log(
            df["hourly_wage_real"].replace(0, np.nan)
        )

    # Create interaction dummies
    df = df.copy()
    dummies = pd.get_dummies(df[interact_col], prefix=interact_col, drop_first=True, dtype=float)
    interactions = dummies.multiply(df["female"], axis=0)
    interactions.columns = [f"female_x_{c}" for c in dummies.columns]

    available_controls = [c for c in controls if c in df.columns]
    feature_cols = ["female"] + available_controls
    X_parts = [df[feature_cols].astype(float), dummies, interactions]
    X = sm.add_constant(pd.concat(X_parts, axis=1))
    y = df[outcome]
    w = df[weight_col]

    valid = y.notna() & w.notna() & X.notna().all(axis=1)
    X, y, w = X[valid], y[valid], w[valid]

    model = sm.WLS(y, X, weights=w).fit()

    # Extract interaction coefficients
    int_cols = [c for c in model.params.index if c.startswith("female_x_")]
    results = pd.DataFrame({
        "interaction": int_cols,
        "coef": model.params[int_cols].values,
        "se": model.bse[int_cols].values,
        "pvalue": model.pvalues[int_cols].values,
    })
    # Add base female coefficient
    base = pd.DataFrame([{
        "interaction": "female (base)",
        "coef": model.params["female"],
        "se": model.bse["female"],
        "pvalue": model.pvalues["female"],
    }])
    return pd.concat([base, results], ignore_index=True)


def _default_dimensions(df: pd.DataFrame) -> list[str]:
    """Return default stratification dimensions from available columns."""
    candidates = [
        "education_level", "race_ethnicity", "marital_status",
        "occupation_broad", "industry_broad",
        "commute_bin", "work_from_home",
        "parenthood_category", "state_fips",
    ]
    return [c for c in candidates if c in df.columns]
