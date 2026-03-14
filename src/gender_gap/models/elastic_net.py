"""Elastic Net penalized model with female x covariate interactions.

Discovers heterogeneous gap patterns via interaction terms
without manual specification of all interactions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ElasticNetResult:
    """Result of Elastic Net wage model."""

    alpha: float
    l1_ratio: float
    female_coef: float
    n_nonzero: int
    n_total: int
    r_squared: float
    n_obs: int
    top_interactions: pd.DataFrame


def run_elastic_net(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    weight_col: str = "person_weight",
    interaction_vars: list[str] | None = None,
    l1_ratios: list[float] | None = None,
    n_alphas: int = 50,
    cv: int = 5,
) -> ElasticNetResult:
    """Fit Elastic Net with female x covariate interactions.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data.
    outcome : str
        Dependent variable.
    weight_col : str
        Sample weights.
    interaction_vars : list[str] | None
        Variables to interact with female. If None, uses defaults.
    l1_ratios : list[float] | None
        Grid of l1_ratio values.
    n_alphas : int
        Number of alpha values in the path.
    cv : int
        Number of cross-validation folds.
    """
    if interaction_vars is None:
        interaction_vars = [
            "age", "usual_hours_week", "work_from_home",
            "commute_minutes_one_way", "number_children",
            "children_under_5",
        ]

    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]

    if outcome not in df.columns and outcome == "log_hourly_wage_real":
        df = df.copy()
        df["log_hourly_wage_real"] = np.log(
            df["hourly_wage_real"].replace(0, np.nan)
        )

    # Build feature matrix
    feature_cols = ["female"]
    base_numeric = [c for c in interaction_vars if c in df.columns]
    feature_cols.extend(base_numeric)

    # Create interactions: female * each numeric var
    interaction_names = []
    for var in base_numeric:
        iname = f"female_x_{var}"
        df[iname] = df["female"] * df[var]
        interaction_names.append(iname)
    feature_cols.extend(interaction_names)

    # Clean
    valid = df[outcome].notna() & df[weight_col].notna()
    for col in feature_cols:
        valid = valid & df[col].notna()
    clean = df[valid].copy()

    if len(clean) < 50:
        raise ValueError(f"Too few valid observations: {len(clean)}")

    X = clean[feature_cols].astype(float).values
    y = clean[outcome].values
    w = clean[weight_col].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit
    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=n_alphas,
        cv=cv,
        max_iter=10000,
    )
    model.fit(X_scaled, y, sample_weight=w)

    # Extract results
    coefs = pd.Series(model.coef_, index=feature_cols)
    female_coef = coefs.get("female", np.nan)
    n_nonzero = int((model.coef_ != 0).sum())

    # R-squared
    y_pred = model.predict(X_scaled)
    ss_res = np.average((y - y_pred) ** 2, weights=w)
    ss_tot = np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    r_sq = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Top interactions by absolute coefficient
    interaction_coefs = coefs[interaction_names].abs().sort_values(ascending=False)
    top_int = pd.DataFrame({
        "interaction": interaction_coefs.index,
        "abs_coef": interaction_coefs.values,
        "coef": coefs[interaction_coefs.index].values,
    }).head(10)

    return ElasticNetResult(
        alpha=model.alpha_,
        l1_ratio=model.l1_ratio_,
        female_coef=female_coef,
        n_nonzero=n_nonzero,
        n_total=len(feature_cols),
        r_squared=r_sq,
        n_obs=len(clean),
        top_interactions=top_int,
    )
