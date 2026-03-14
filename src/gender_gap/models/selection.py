"""Employment-selection robustness models.

Implements a transparent two-part annual-earnings design plus an IPW-style
worker-wage sensitivity check. This is intended as a robustness layer, not a
claim that selection is fully identified.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from gender_gap.models.ols import _fit_weighted_least_squares
from gender_gap.utils.weights import weighted_mean

logger = logging.getLogger(__name__)


SELECTION_BLOCKS = {
    "S0": ["female"],
    "S1": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
    "S2": [
        "female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
        "C(state_fips)", "C(marital_status)", "number_children", "children_under_5",
    ],
}


def run_selection_robustness(
    df: pd.DataFrame,
    blocks: dict[str, list[str]] | None = None,
    weight_col: str = "person_weight",
    employed_col: str = "employed",
    annual_earnings_col: str = "annual_earnings_real",
    wage_col: str = "hourly_wage_real",
    ipw_clip_min: float = 0.02,
) -> pd.DataFrame:
    """Run two-part earnings and IPW wage-gap robustness models."""
    if blocks is None:
        blocks = SELECTION_BLOCKS

    results = []
    for model_name, controls in blocks.items():
        result = _run_selection_block(
            df=df,
            model_name=model_name,
            controls=controls,
            weight_col=weight_col,
            employed_col=employed_col,
            annual_earnings_col=annual_earnings_col,
            wage_col=wage_col,
            ipw_clip_min=ipw_clip_min,
        )
        results.append(result)

    return pd.DataFrame(results)


def _run_selection_block(
    df: pd.DataFrame,
    model_name: str,
    controls: list[str],
    weight_col: str,
    employed_col: str,
    annual_earnings_col: str,
    wage_col: str,
    ipw_clip_min: float,
) -> dict[str, float]:
    model_df = df.copy()
    model_df = model_df[
        model_df[weight_col].notna()
        & (model_df[weight_col] > 0)
        & model_df[employed_col].notna()
    ].copy()
    model_df["log_annual_earnings_real"] = np.nan
    positive_earnings = model_df[annual_earnings_col] > 0
    model_df.loc[positive_earnings, "log_annual_earnings_real"] = np.log(
        model_df.loc[positive_earnings, annual_earnings_col]
    )

    X_full = _build_controls_matrix(model_df, controls)
    model_df = model_df.loc[X_full.index].copy()
    weights = model_df[weight_col].astype(float)

    y_emp = model_df[employed_col].astype(float)
    employment_fit = _fit_weighted_binomial(y_emp, X_full, weights)

    p_actual = np.clip(employment_fit.predict(X_full), ipw_clip_min, 1.0)
    X_male = X_full.copy()
    X_female = X_full.copy()
    if "female" in X_male.columns:
        X_male["female"] = 0.0
        X_female["female"] = 1.0
    p_male = np.clip(employment_fit.predict(X_male), ipw_clip_min, 1.0)
    p_female = np.clip(employment_fit.predict(X_female), ipw_clip_min, 1.0)

    employed_mask = (
        (model_df[employed_col] == 1)
        & model_df[annual_earnings_col].gt(0)
        & model_df["log_annual_earnings_real"].notna()
    )
    if employed_mask.sum() < 10:
        raise ValueError(f"{model_name}: too few employed observations")

    y_earn = model_df.loc[employed_mask, "log_annual_earnings_real"]
    X_earn = X_full.loc[employed_mask]
    w_earn = weights.loc[employed_mask]
    earnings_fit = _fit_weighted_least_squares(y_earn, X_earn, w_earn)

    fitted_log = X_earn.to_numpy(dtype=float) @ earnings_fit["params"].to_numpy(dtype=float)
    smear = weighted_mean(np.exp(y_earn.to_numpy(dtype=float) - fitted_log), w_earn.to_numpy(dtype=float))

    cond_male = np.exp(X_male.to_numpy(dtype=float) @ earnings_fit["params"].to_numpy(dtype=float)) * smear
    cond_female = np.exp(X_female.to_numpy(dtype=float) @ earnings_fit["params"].to_numpy(dtype=float)) * smear
    expected_male = p_male * cond_male
    expected_female = p_female * cond_female

    total_male_mean = weighted_mean(
        model_df.loc[model_df["female"] == 0, annual_earnings_col],
        weights.loc[model_df["female"] == 0],
    )
    total_female_mean = weighted_mean(
        model_df.loc[model_df["female"] == 1, annual_earnings_col],
        weights.loc[model_df["female"] == 1],
    )

    worker_mask = (
        (model_df[employed_col] == 1)
        & model_df[wage_col].notna()
        & model_df[wage_col].gt(0)
    )
    observed_worker_gap_pct = np.nan
    ipw_worker_gap_pct = np.nan
    if worker_mask.any():
        worker_df = model_df.loc[worker_mask].copy()
        worker_weights = weights.loc[worker_mask]
        p_worker = p_actual[worker_mask.to_numpy()]
        observed_worker_gap_pct = _gap_pct(
            worker_df.loc[worker_df["female"] == 0, wage_col],
            worker_weights.loc[worker_df["female"] == 0],
            worker_df.loc[worker_df["female"] == 1, wage_col],
            worker_weights.loc[worker_df["female"] == 1],
        )
        ipw_weights = worker_weights / p_worker
        ipw_worker_gap_pct = _gap_pct(
            worker_df.loc[worker_df["female"] == 0, wage_col],
            ipw_weights.loc[worker_df["female"] == 0],
            worker_df.loc[worker_df["female"] == 1, wage_col],
            ipw_weights.loc[worker_df["female"] == 1],
        )

    expected_male_mean = weighted_mean(expected_male, weights)
    expected_female_mean = weighted_mean(expected_female, weights)

    return {
        "model": model_name,
        "employment_female_prob_pp": weighted_mean((p_female - p_male) * 100.0, weights),
        "female_conditional_log_earnings_coef": float(earnings_fit["params"]["female"]),
        "female_conditional_earnings_pct": (math.exp(float(earnings_fit["params"]["female"])) - 1.0) * 100.0,
        "combined_expected_earnings_male": expected_male_mean,
        "combined_expected_earnings_female": expected_female_mean,
        "combined_expected_earnings_gap_dollars": expected_male_mean - expected_female_mean,
        "combined_expected_earnings_gap_pct": _gap_pct_from_means(expected_male_mean, expected_female_mean),
        "observed_total_earnings_gap_pct": _gap_pct_from_means(total_male_mean, total_female_mean),
        "observed_worker_hourly_gap_pct": observed_worker_gap_pct,
        "ipw_worker_hourly_gap_pct": ipw_worker_gap_pct,
        "employment_model_female_coef": float(employment_fit.params["female"]),
        "n_total": int(len(model_df)),
        "n_employed": int(employed_mask.sum()),
    }


def _build_controls_matrix(df: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    """Build a design matrix from the shared formula-style control syntax."""
    x_parts = []
    for term in controls:
        cat_match = re.match(r"C\((\w+)\)", term)
        if cat_match:
            col = cat_match.group(1)
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
                x_parts.append(dummies)
        elif term in df.columns:
            x_parts.append(df[[term]].astype(float))

    if x_parts:
        X = pd.concat(x_parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)

    X = sm.add_constant(X)
    valid = X.notna().all(axis=1)
    return X.loc[valid].copy()


def _fit_weighted_binomial(
    y: pd.Series,
    X: pd.DataFrame,
    weights: pd.Series,
):
    """Fit a weighted binomial model, falling back to LPM if needed."""
    y = y.loc[X.index]
    weights = weights.loc[X.index]
    try:
        model = sm.GLM(
            y,
            X,
            family=sm.families.Binomial(),
            freq_weights=weights,
        )
        return model.fit(maxiter=100, disp=0)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("Binomial selection model failed, falling back to LPM: %s", exc)
        fit = _fit_weighted_least_squares(y, X, weights)

        class _LPMResult:
            params = fit["params"]

            @staticmethod
            def predict(X_new):
                arr = X_new.to_numpy(dtype=float) @ fit["params"].to_numpy(dtype=float)
                return np.clip(arr, 0.001, 0.999)

        return _LPMResult()


def _gap_pct(male_values, male_weights, female_values, female_weights) -> float:
    male_mean = weighted_mean(male_values, male_weights)
    female_mean = weighted_mean(female_values, female_weights)
    return _gap_pct_from_means(male_mean, female_mean)


def _gap_pct_from_means(male_mean: float, female_mean: float) -> float:
    if male_mean == 0 or not np.isfinite(male_mean) or not np.isfinite(female_mean):
        return np.nan
    return (male_mean - female_mean) / male_mean * 100.0
