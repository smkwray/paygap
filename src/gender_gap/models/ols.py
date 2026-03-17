"""Sequential survey-weighted OLS models.

Implements the M0-M8 progressive control specification:
- M0: female only
- M1: + age, race_ethnicity, education
- M2: + geography + local context
- M3: + industry + occupation + class_of_worker
- M4: + hours + overtime + commute + WFH
- M5: + marital status + children
- M6: + reproductive-burden controls
- M7: + O*NET occupational context
- M8: + slim reproductive x job-context interactions

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
from scipy import sparse

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


@dataclass
class SparseDesignMatrix:
    """Sparse design matrix wrapper that preserves row/column labels."""

    matrix: sparse.csr_matrix
    columns: pd.Index
    index: pd.Index

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Return a float-typed numeric Series with invalid values set to NaN."""
    return pd.to_numeric(series, errors="coerce")


# ACS pooled and by-year panels both become prohibitively expensive as dense
# dummy matrices well below one million rows once occupation and industry FEs
# are expanded.
SPARSE_DESIGN_ROW_THRESHOLD = 250_000


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
    "M6_reproductive": [
        "female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
        "C(state_fips)", "C(occupation_code)", "C(industry_code)",
        "C(class_of_worker)", "usual_hours_week", "work_from_home",
        "commute_minutes_one_way", "C(marital_status)", "number_children",
        "children_under_5", "recent_birth", "recent_marriage", "has_own_child",
        "own_child_under6", "own_child_6_17_only", "C(couple_type)",
        "C(reproductive_stage)",
    ],
    "M7_onet_context": [
        "female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
        "C(state_fips)", "C(occupation_code)", "C(industry_code)",
        "C(class_of_worker)", "usual_hours_week", "work_from_home",
        "commute_minutes_one_way", "C(marital_status)", "number_children",
        "children_under_5", "recent_birth", "recent_marriage", "has_own_child",
        "own_child_under6", "own_child_6_17_only", "C(couple_type)",
        "C(reproductive_stage)", "autonomy", "schedule_unpredictability",
        "time_pressure", "coordination_responsibility", "physical_proximity",
        "job_rigidity",
    ],
    "M8_reproductive_x_job_context": [
        "female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
        "C(state_fips)", "C(occupation_code)", "C(industry_code)",
        "C(class_of_worker)", "usual_hours_week", "work_from_home",
        "commute_minutes_one_way", "C(marital_status)", "number_children",
        "children_under_5", "recent_birth", "recent_marriage", "has_own_child",
        "own_child_under6", "own_child_6_17_only", "C(couple_type)",
        "C(reproductive_stage)", "autonomy", "schedule_unpredictability",
        "time_pressure", "coordination_responsibility", "physical_proximity",
        "job_rigidity", "female_x_recent_birth", "female_x_own_child_under6",
        "female_x_recent_marriage", "female_x_same_sex_couple_household",
        "female_x_autonomy", "female_x_job_rigidity",
        "female_x_own_child_under6_x_job_rigidity",
    ],
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
        hourly = _coerce_numeric_series(df["hourly_wage_real"])
        df["log_hourly_wage_real"] = np.log(hourly.where(hourly != 0))

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
        if _design_nobs(X) < 10:
            logger.warning("%s: too few observations (%d)", model_name, _design_nobs(X))
            return None
        if "female" not in _design_columns(X):
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
    if outcome not in df.columns and outcome == "log_hourly_wage_real":
        df = df.copy()
        hourly = _coerce_numeric_series(df["hourly_wage_real"])
        df["log_hourly_wage_real"] = np.log(hourly.where(hourly != 0))

    formula = f"{outcome} ~ " + " + ".join(controls)
    outcome_values = _coerce_numeric_series(df[outcome])
    weight_values = _coerce_numeric_series(df[weight_col])
    mask = outcome_values.notna() & weight_values.notna() & (weight_values > 0)
    analysis_df = df.loc[mask].copy()
    analysis_df[outcome] = outcome_values.loc[mask]
    analysis_df[weight_col] = weight_values.loc[mask]
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
    w_arr = weights.to_numpy(dtype=float, copy=False)

    if isinstance(X, SparseDesignMatrix):
        X_arr = X.matrix
        valid = np.isfinite(y_arr) & np.isfinite(w_arr) & (w_arr > 0)
        y_arr = y_arr[valid]
        w_arr = w_arr[valid]
        X_arr = X_arr[valid]
        if X_arr.shape[0] == 0:
            raise ValueError("No observations with valid positive weights")

        sqrt_w = np.sqrt(w_arr)
        X_w = X_arr.multiply(sqrt_w[:, None])
        y_w = y_arr * sqrt_w

        xtwx = (X_w.T @ X_w).toarray()
        xtwy = np.asarray(X_w.T @ y_w).ravel()
        beta, _, _, _ = np.linalg.lstsq(xtwx, xtwy, rcond=None)

        fitted = np.asarray(X_arr @ beta).ravel()
        resid = y_arr - fitted
        sse = float(np.sum(w_arr * resid ** 2))

        n_obs, n_params = X_arr.shape
        dof = max(n_obs - n_params, 1)
        sigma2 = sse / dof

        xtwx_inv = np.linalg.pinv(xtwx)
        index = X.columns
    else:
        X_arr = X.to_numpy(dtype=float, copy=False)
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
        index = X.columns

    cov = sigma2 * xtwx_inv
    se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))

    t_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvalues = 2 * stats.t.sf(np.abs(t_stats), df=dof)

    weight_sum = float(np.sum(w_arr))
    y_mean = float(np.sum(w_arr * y_arr) / weight_sum)
    sst = float(np.sum(w_arr * (y_arr - y_mean) ** 2))
    r_squared = 1.0 - (sse / sst) if sst > 0 else float("nan")

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
) -> tuple[pd.Series, pd.DataFrame | SparseDesignMatrix]:
    """Build y and X from a patsy-style formula string.

    Handles C() categorical markers by creating dummies.
    """
    parts = formula.split("~")
    outcome_col = parts[0].strip()
    rhs = parts[1].strip()

    y = _coerce_numeric_series(df[outcome_col])

    # Parse RHS terms
    terms = [t.strip() for t in rhs.split("+")]
    if len(df) >= SPARSE_DESIGN_ROW_THRESHOLD:
        return _build_sparse_design_matrix(df, y, terms)

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


def _build_sparse_design_matrix(
    df: pd.DataFrame,
    y: pd.Series,
    terms: list[str],
) -> tuple[pd.Series, SparseDesignMatrix]:
    valid = y.notna().copy()
    numeric_terms: list[str] = []
    categorical_terms: list[str] = []

    for term in terms:
        cat_match = re.match(r"C\((\w+)\)", term)
        if cat_match:
            col = cat_match.group(1)
            if col in df.columns:
                categorical_terms.append(col)
        elif term in df.columns:
            values = _coerce_numeric_series(df[term])
            valid &= values.notna()
            numeric_terms.append(term)

    design_df = df.loc[valid]
    y_valid = y.loc[valid]
    n_obs = len(design_df)

    matrices: list[sparse.csr_matrix] = [
        sparse.csr_matrix(np.ones((n_obs, 1), dtype=float))
    ]
    columns = ["const"]

    for term in numeric_terms:
        values = _coerce_numeric_series(design_df[term]).to_numpy(dtype=float, copy=False)
        matrices.append(sparse.csr_matrix(values.reshape(-1, 1)))
        columns.append(term)

    for col in categorical_terms:
        categorical = pd.Categorical(design_df[col])
        if len(categorical.categories) <= 1:
            continue
        codes = categorical.codes
        valid_codes = codes >= 1
        if not np.any(valid_codes):
            continue

        row_idx = np.nonzero(valid_codes)[0]
        col_idx = codes[valid_codes] - 1
        data = np.ones(len(row_idx), dtype=float)
        dummy_matrix = sparse.coo_matrix(
            (data, (row_idx, col_idx)),
            shape=(n_obs, len(categorical.categories) - 1),
        ).tocsr()
        matrices.append(dummy_matrix)
        columns.extend(
            [f"{col}_{category}" for category in categorical.categories[1:]]
        )

    matrix = sparse.hstack(matrices, format="csr")
    return y_valid, SparseDesignMatrix(matrix=matrix, columns=pd.Index(columns), index=design_df.index)


def _design_nobs(X: pd.DataFrame | SparseDesignMatrix) -> int:
    return int(X.shape[0]) if isinstance(X, SparseDesignMatrix) else len(X)


def _design_columns(X: pd.DataFrame | SparseDesignMatrix) -> pd.Index:
    return X.columns


def _design_index(X: pd.DataFrame | SparseDesignMatrix) -> pd.Index:
    return X.index


def _design_matmul(X: pd.DataFrame | SparseDesignMatrix, beta: np.ndarray) -> np.ndarray:
    if isinstance(X, SparseDesignMatrix):
        return np.asarray(X.matrix @ beta).ravel()
    return X.to_numpy(dtype=float, copy=False) @ beta


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
            "controls": " + ".join(r.controls),
        })
    return pd.DataFrame(rows)


def coefficient_table(
    df: pd.DataFrame,
    model_name: str,
    outcome: str = "log_hourly_wage_real",
    weight_col: str = "person_weight",
    blocks: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Fit a named design and return coefficients for all included terms."""
    if blocks is None:
        blocks = BLOCK_DEFINITIONS
    controls = blocks.get(model_name)
    if controls is None:
        raise ValueError(f"Unknown model_name: {model_name}")

    y, X, weights = _prepare_ols_inputs(df, outcome, controls, weight_col)
    fit = _fit_weighted_least_squares(y, X, weights)
    return pd.DataFrame(
        {
            "model": model_name,
            "term": fit["params"].index,
            "coef": fit["params"].values,
            "se": fit["bse"].values,
            "pvalue": fit["pvalues"].values,
            "n_obs": fit["n_obs"],
            "r_squared": fit["r_squared"],
        }
    )


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
    if _design_nobs(X) < 10:
        raise ValueError(f"Too few observations for model {model_name}")
    if "female" not in _design_columns(X):
        raise ValueError("'female' not in design matrix")

    fit = _fit_weighted_least_squares(y, X, weights)

    rep_cols = replicate_weight_columns(df.columns, prefix=repweight_prefix)
    if not rep_cols:
        raise ValueError("No ACS replicate-weight columns found")

    rep_estimates = []
    for idx, rep_col in enumerate(rep_cols, start=1):
        rep_weights = df.loc[_design_index(X), rep_col]
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
