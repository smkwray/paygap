"""Sequential survey-weighted OLS models.

Implements the M0-M6 progressive control specification:
- M0: female only
- M1: + age, race_ethnicity, education
- M2: + geography + local context
- M3: + industry + occupation + class_of_worker
- M4: + hours + overtime + commute + WFH
- M5: + marital status + children

NLSY-specific blocks add g_proxy (cognitive ability) as M_g variants.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from gender_gap.utils.weights import replicate_weight_columns, sdr_summary

logger = logging.getLogger(__name__)


@dataclass
class OLSResult:
    """Result of a single weighted OLS regression."""

    model_name: str
    female_coef: float
    female_se: float
    female_pvalue: float
    r_squared: float
    n_obs: int
    controls: list[str]


# Control blocks for sequential models
BLOCK_DEFINITIONS = {
    "M0": ["female"],
    "M1": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
    "M2": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "C(state_fips)"],
    "M3": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "C(state_fips)", "C(occupation_code)", "C(industry_code)",
            "C(class_of_worker)"],
    "M4": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "C(state_fips)", "C(occupation_code)", "C(industry_code)",
            "C(class_of_worker)",
            "usual_hours_week", "work_from_home", "commute_minutes_one_way"],
    "M5": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "C(state_fips)", "C(occupation_code)", "C(industry_code)",
            "C(class_of_worker)",
            "usual_hours_week", "work_from_home", "commute_minutes_one_way",
            "C(marital_status)", "number_children", "children_under_5"],
}

# NLSY-specific blocks: add cognitive ability (g_proxy) from ASVAB/AFQT
# These are used when running models on NLSY data with g as a control
NLSY_BLOCK_DEFINITIONS = {
    "N0": ["female"],
    "N1": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
    "N2": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "g_proxy"],
    "N3": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "g_proxy", "C(occupation_code)"],
    "N4": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "g_proxy", "C(occupation_code)",
            "C(marital_status)", "number_children"],
    "N5": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "g_proxy", "C(occupation_code)",
            "C(marital_status)", "number_children",
            "parent_education"],
}


def run_sequential_ols(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    weight_col: str = "person_weight",
    blocks: dict[str, list[str]] | None = None,
) -> list[OLSResult]:
    """Run the M0-M5 sequential OLS models.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready DataFrame with outcome and all control variables.
    outcome : str
        Dependent variable column name.
    weight_col : str
        Survey weight column.
    blocks : dict | None
        Custom block definitions. If None, uses BLOCK_DEFINITIONS.

    Returns
    -------
    list[OLSResult]
    """
    if blocks is None:
        blocks = BLOCK_DEFINITIONS

    if outcome not in df.columns and outcome == "log_hourly_wage_real":
        df = df.copy()
        df["log_hourly_wage_real"] = np.log(
            df["hourly_wage_real"].replace(0, np.nan)
        )

    results = []
    for model_name, controls in blocks.items():
        result = _run_single_ols(df, outcome, controls, weight_col, model_name)
        if result is not None:
            results.append(result)
            logger.info(
                "%s: female=%.4f (SE=%.4f), R²=%.4f, n=%d",
                model_name, result.female_coef, result.female_se,
                result.r_squared, result.n_obs,
            )

    return results


def _run_single_ols(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    weight_col: str,
    model_name: str,
) -> OLSResult | None:
    """Run a single WLS model and extract the female coefficient."""
    try:
        y, X, weights = _prepare_ols_inputs(df, outcome, controls, weight_col)
        if len(X) < 10:
            logger.warning("%s: too few observations (%d)", model_name, len(X))
            return None
        if "female" not in X.columns:
            logger.warning("%s: 'female' not in design matrix", model_name)
            return None

        fit = _fit_weighted_least_squares(y, X, weights)

        return OLSResult(
            model_name=model_name,
            female_coef=fit["params"]["female"],
            female_se=fit["bse"]["female"],
            female_pvalue=fit["pvalues"]["female"],
            r_squared=fit["r_squared"],
            n_obs=fit["n_obs"],
            controls=controls,
        )
    except Exception as e:
        logger.warning("%s failed: %s", model_name, e)
        return None


def _prepare_ols_inputs(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    weight_col: str,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Build a common OLS design and aligned weight vector once."""
    formula = f"{outcome} ~ " + " + ".join(controls)
    mask = df[outcome].notna() & df[weight_col].notna() & (df[weight_col] > 0)
    analysis_df = df.loc[mask].copy()
    y, X = _build_design_matrix(analysis_df, formula)
    weights = analysis_df.loc[X.index, weight_col]
    return y, X, weights


def _fit_weighted_least_squares(
    y: pd.Series,
    X: pd.DataFrame,
    weights: pd.Series,
) -> dict:
    """Fit WLS via weighted least squares on the transformed design.

    This avoids the overhead of building a full statsmodels result object for
    every sequential model while preserving the coefficient, classical SE, and
    p-value outputs used elsewhere in the repo.
    """
    y_arr = y.to_numpy(dtype=float, copy=False)
    X_arr = X.to_numpy(dtype=float, copy=False)
    w_arr = weights.to_numpy(dtype=float, copy=False)

    valid = np.isfinite(y_arr) & np.isfinite(w_arr) & (w_arr > 0)
    valid &= np.isfinite(X_arr).all(axis=1)
    y_arr = y_arr[valid]
    X_arr = X_arr[valid]
    w_arr = w_arr[valid]
    if X_arr.shape[0] == 0:
        raise ValueError("No observations with valid positive weights")

    sqrt_w = np.sqrt(w_arr)
    X_w = X_arr * sqrt_w[:, None]
    y_w = y_arr * sqrt_w

    beta, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)

    fitted = X_arr @ beta
    resid = y_arr - fitted
    sse = float(np.sum(w_arr * resid ** 2))

    n_obs, n_params = X_arr.shape
    dof = max(n_obs - n_params, 1)
    sigma2 = sse / dof

    xtwx_inv = np.linalg.pinv(X_w.T @ X_w)
    cov = sigma2 * xtwx_inv
    se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))

    t_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvalues = 2 * stats.t.sf(np.abs(t_stats), df=dof)

    weight_sum = float(np.sum(w_arr))
    y_mean = float(np.sum(w_arr * y_arr) / weight_sum)
    sst = float(np.sum(w_arr * (y_arr - y_mean) ** 2))
    r_squared = 1.0 - (sse / sst) if sst > 0 else float("nan")

    index = X.columns
    return {
        "params": pd.Series(beta, index=index),
        "bse": pd.Series(se, index=index),
        "pvalues": pd.Series(pvalues, index=index),
        "r_squared": r_squared,
        "n_obs": int(n_obs),
    }


def _build_design_matrix(
    df: pd.DataFrame,
    formula: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build y and X from a patsy-style formula string.

    Handles C() categorical markers by creating dummies.
    """
    parts = formula.split("~")
    outcome_col = parts[0].strip()
    rhs = parts[1].strip()

    y = df[outcome_col]

    # Parse RHS terms
    terms = [t.strip() for t in rhs.split("+")]
    X_parts = []
    for term in terms:
        cat_match = re.match(r"C\((\w+)\)", term)
        if cat_match:
            col = cat_match.group(1)
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
                X_parts.append(dummies)
        elif term in df.columns:
            X_parts.append(df[[term]].astype(float))

    if X_parts:
        X = pd.concat(X_parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)

    X = sm.add_constant(X)

    # Drop rows with any NaN
    valid = y.notna() & X.notna().all(axis=1)
    return y[valid], X[valid]


def design_source_columns(controls: list[str]) -> list[str]:
    """Return raw DataFrame columns needed to build a model design."""
    columns: list[str] = []
    for term in controls:
        cat_match = re.match(r"C\((\w+)\)", term)
        col = cat_match.group(1) if cat_match else term
        if col not in columns:
            columns.append(col)
    return columns


def required_columns_for_model(
    model_name: str,
    outcome: str = "log_hourly_wage_real",
    weight_col: str = "person_weight",
    blocks: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return the minimum columns needed to fit a named sequential model."""
    if blocks is None:
        blocks = BLOCK_DEFINITIONS
    controls = blocks.get(model_name)
    if controls is None:
        raise ValueError(f"Unknown model_name: {model_name}")

    required = [outcome, weight_col]
    for column in design_source_columns(controls):
        if column not in required:
            required.append(column)
    return required


def results_to_dataframe(results: list[OLSResult]) -> pd.DataFrame:
    """Convert list of OLSResult to a summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "model": r.model_name,
            "female_coef": r.female_coef,
            "female_se": r.female_se,
            "female_pvalue": r.female_pvalue,
            "r_squared": r.r_squared,
            "n_obs": r.n_obs,
        })
    return pd.DataFrame(rows)


def female_coefficient_with_sdr(
    df: pd.DataFrame,
    model_name: str = "M5",
    outcome: str = "log_hourly_wage_real",
    weight_col: str = "person_weight",
    repweight_prefix: str = "PWGTP",
    blocks: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Compute ACS SDR uncertainty for a model's female coefficient."""
    if blocks is None:
        blocks = BLOCK_DEFINITIONS
    controls = blocks.get(model_name)
    if controls is None:
        raise ValueError(f"Unknown model_name: {model_name}")
    y, X, weights = _prepare_ols_inputs(df, outcome, controls, weight_col)
    if len(X) < 10:
        raise ValueError(f"Too few observations for model {model_name}")
    if "female" not in X.columns:
        raise ValueError("'female' not in design matrix")

    fit = _fit_weighted_least_squares(y, X, weights)

    rep_cols = replicate_weight_columns(df.columns, prefix=repweight_prefix)
    if not rep_cols:
        raise ValueError("No ACS replicate-weight columns found")

    rep_estimates = []
    for idx, rep_col in enumerate(rep_cols, start=1):
        rep_weights = df.loc[X.index, rep_col]
        rep_fit = _fit_weighted_least_squares(y, X, rep_weights)
        rep_estimates.append(rep_fit["params"]["female"])
        if idx % 10 == 0 or idx == len(rep_cols):
            logger.info(
                "%s SDR replicates: %d/%d complete",
                model_name,
                idx,
                len(rep_cols),
            )

    summary = sdr_summary(fit["params"]["female"], rep_estimates)
    return {
        "model": model_name,
        "female_coef": fit["params"]["female"],
        "female_coef_classical_se": fit["bse"]["female"],
        "female_coef_sdr_se": summary["se"],
        "female_coef_moe90": summary["moe90"],
        "female_coef_ci90_low": summary["ci90_low"],
        "female_coef_ci90_high": summary["ci90_high"],
        "female_coef_ci95_low": summary["ci95_low"],
        "female_coef_ci95_high": summary["ci95_high"],
        "female_pvalue": fit["pvalues"]["female"],
        "r_squared": fit["r_squared"],
        "n_obs": fit["n_obs"],
        "n_replicates": len(rep_estimates),
    }
