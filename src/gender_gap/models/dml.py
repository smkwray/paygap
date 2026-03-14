"""Double Machine Learning (DML) for adjusted gender gap estimation.

Uses partial-linear DML with `female` as treatment and
`log(hourly_wage_real)` as outcome. Flexible nuisance models
handle high-dimensional controls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DMLResult:
    """Result of a DoubleML adjusted gap estimation."""

    treatment_effect: float
    std_error: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    n_obs: int
    nuisance_learner: str


def run_dml(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    treatment: str = "female",
    weight_col: str = "person_weight",
    controls: list[str] | None = None,
    nuisance_learner: str = "lgbm",
    n_folds: int = 5,
    random_seed: int = 42,
) -> DMLResult:
    """Run partial-linear DoubleML to estimate the adjusted gender gap.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data.
    outcome : str
        Dependent variable.
    treatment : str
        Treatment variable (binary female indicator).
    weight_col : str
        Sample weight column name. Currently retained for API consistency,
        but the present DoubleML implementation does not use survey weights.
    controls : list[str] | None
        Control variables for nuisance models.
    nuisance_learner : str
        'lgbm', 'rf', or 'elasticnet'.
    n_folds : int
        Number of cross-fitting folds.
    random_seed : int
        Random seed.
    """
    import doubleml as dml
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV

    if controls is None:
        controls = _default_controls(df)

    if outcome not in df.columns and outcome == "log_hourly_wage_real":
        df = df.copy()
        df["log_hourly_wage_real"] = np.log(
            df["hourly_wage_real"].replace(0, np.nan)
        )

    # Clean
    all_cols = [outcome, treatment] + controls
    valid = pd.Series(True, index=df.index)
    for col in all_cols:
        if col in df.columns:
            valid = valid & df[col].notna()
    clean = df[valid].copy()

    if len(clean) < 100:
        raise ValueError(f"Too few valid observations for DML: {len(clean)}")

    X = clean[controls].copy()
    categorical = [col for col in controls if not pd.api.types.is_numeric_dtype(X[col])]
    if categorical:
        X = pd.get_dummies(X, columns=categorical, drop_first=True, dtype=float)
    else:
        X = X.astype(float)

    dml_frame = pd.concat([clean[[outcome, treatment]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    # Build DoubleML data object
    dml_data = dml.DoubleMLData(
        dml_frame,
        y_col=outcome,
        d_cols=treatment,
        x_cols=list(X.columns),
    )

    # Select nuisance learner
    if nuisance_learner == "lgbm":
        ml_l = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=random_seed,
        )
        ml_m = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=random_seed,
        )
    elif nuisance_learner == "rf":
        ml_l = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=random_seed,
        )
        ml_m = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=random_seed,
        )
    elif nuisance_learner == "elasticnet":
        ml_l = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000)
        ml_m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000)
    else:
        raise ValueError(f"Unknown nuisance_learner: {nuisance_learner}")

    # Fit partial linear model
    dml_plr = dml.DoubleMLPLR(
        dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=n_folds,
    )
    dml_plr.fit()

    # Extract results
    summary = dml_plr.summary
    coef = float(summary["coef"].iloc[0])
    se = float(summary["std err"].iloc[0])
    pval = float(summary["P>|t|"].iloc[0])
    ci = dml_plr.confint()
    ci_lo = float(ci.iloc[0, 0])
    ci_hi = float(ci.iloc[0, 1])

    return DMLResult(
        treatment_effect=coef,
        std_error=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        pvalue=pval,
        n_obs=len(dml_frame),
        nuisance_learner=nuisance_learner,
    )


def _default_controls(df: pd.DataFrame) -> list[str]:
    """Return default control variable list from available columns."""
    candidates = [
        "age", "age_sq",
        "race_ethnicity", "education_level",
        "marital_status", "number_children", "children_under_5",
        "occupation_code", "industry_code", "class_of_worker",
        "usual_hours_week", "work_from_home",
        "commute_minutes_one_way",
        "state_fips",
    ]
    return [c for c in candidates if c in df.columns]
