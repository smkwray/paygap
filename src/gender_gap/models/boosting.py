"""Tree boosting model for nonlinear wage prediction.

Uses CatBoost (default) or LightGBM for nonlinear prediction
and SHAP-style feature importance diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BoostingResult:
    """Result of a boosting wage model."""

    library: str
    rmse: float
    r_squared: float
    n_obs: int
    feature_importance: pd.DataFrame
    female_gap_partial: float | None


def run_catboost(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    weight_col: str = "person_weight",
    features: list[str] | None = None,
    cat_features: list[str] | None = None,
    test_size: float = 0.2,
    iterations: int = 500,
    random_seed: int = 42,
) -> BoostingResult:
    """Fit CatBoost regressor for wage prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data.
    outcome : str
        Target variable.
    weight_col : str
        Sample weights.
    features : list[str] | None
        Feature columns. If None, uses defaults.
    cat_features : list[str] | None
        Categorical feature names for CatBoost native handling.
    test_size : float
        Fraction of data for evaluation.
    iterations : int
        Number of boosting rounds.
    random_seed : int
        Random seed.
    """
    from catboost import CatBoostRegressor, Pool

    if features is None:
        features = _default_features(df)

    if cat_features is None:
        cat_features = [
            c for c in ["race_ethnicity", "education_level", "marital_status",
                         "commute_mode", "class_of_worker"]
            if c in features
        ]

    if outcome not in df.columns and outcome == "log_hourly_wage_real":
        df = df.copy()
        df["log_hourly_wage_real"] = np.log(
            df["hourly_wage_real"].replace(0, np.nan)
        )

    # Clean
    valid = df[outcome].notna() & df[weight_col].notna()
    for col in features:
        if col not in df.columns:
            features = [f for f in features if f != col]
    clean = df[valid].copy()

    # Fill NaN in categoricals
    for c in cat_features:
        if c in clean.columns:
            clean[c] = clean[c].fillna("missing").astype(str)

    # Train/test split
    np.random.seed(random_seed)
    mask = np.random.rand(len(clean)) < (1 - test_size)
    train = clean[mask]
    test = clean[~mask]

    cat_idx = [features.index(c) for c in cat_features if c in features]

    pool_train = Pool(
        train[features], train[outcome],
        weight=train[weight_col],
        cat_features=cat_idx,
    )
    pool_test = Pool(
        test[features], test[outcome],
        weight=test[weight_col],
        cat_features=cat_idx,
    )

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        random_seed=random_seed,
        verbose=0,
    )
    model.fit(pool_train, eval_set=pool_test)

    # Evaluate
    y_pred = model.predict(pool_test)
    y_true = test[outcome].values
    w = test[weight_col].values
    rmse = np.sqrt(np.average((y_true - y_pred) ** 2, weights=w))
    ss_res = np.average((y_true - y_pred) ** 2, weights=w)
    ss_tot = np.average((y_true - np.average(y_true, weights=w)) ** 2, weights=w)
    r_sq = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Feature importance
    importance = model.get_feature_importance()
    fi = pd.DataFrame({
        "feature": features,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    # Partial dependence for female
    gap = None
    if "female" in features:
        test_m = test.copy()
        test_m["female"] = 0
        test_f = test.copy()
        test_f["female"] = 1
        for c in cat_features:
            if c in test_m.columns:
                test_m[c] = test_m[c].astype(str)
                test_f[c] = test_f[c].astype(str)
        pred_m = model.predict(test_m[features])
        pred_f = model.predict(test_f[features])
        gap = float(np.average(pred_m - pred_f, weights=w))

    return BoostingResult(
        library="catboost",
        rmse=rmse,
        r_squared=r_sq,
        n_obs=len(clean),
        feature_importance=fi,
        female_gap_partial=gap,
    )


def _default_features(df: pd.DataFrame) -> list[str]:
    """Return default feature list from available columns."""
    candidates = [
        "female", "age", "age_sq",
        "race_ethnicity", "education_level",
        "marital_status", "number_children", "children_under_5",
        "occupation_code", "industry_code", "class_of_worker",
        "usual_hours_week", "work_from_home",
        "commute_minutes_one_way", "commute_mode",
        "state_fips",
    ]
    return [c for c in candidates if c in df.columns]
