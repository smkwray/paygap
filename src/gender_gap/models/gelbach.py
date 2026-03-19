"""Gelbach (2016) exact decomposition of the gender gap.

The sequential OLS ladder is informative but order-sensitive: the amount
each block "explains" depends on when it enters.  Gelbach's method solves
this by computing how much each covariate block moves the coefficient of
interest (here, female) in an *order-invariant* way.

The identity is exact:

    beta_base - beta_full = sum_k  delta_k

where delta_k is the contribution of covariate block *k*, computed as the
product of the auxiliary-regression coefficients (regressing block-k
covariates on female) and the full-model coefficients on those covariates.

Reference
---------
Gelbach, J. B. (2016). When do covariates matter? And which ones, and how
much? *Journal of Labor Economics*, 34(2), 509-543.

Limitations
-----------
- **Standard errors are approximate.** The delta-method SE for each block
  ignores covariance between the auxiliary-regression gammas and the
  full-model betas.  Treat them as rough guides, not exact inferential
  quantities.
- **Dense design matrices only.** This module does not use sparse matrices.
  For the full 9.7M-row pooled ACS panel, either subsample or run
  year-by-year.  A ``max_rows`` guard raises an error above 2M rows by
  default to prevent silent OOM.

Usage
-----
This module is *additive* — it does not modify any existing pipeline code.
It takes a DataFrame and block definitions and produces a decomposition
table.  Can be called standalone or from a reporting script.

    from gender_gap.models.gelbach import gelbach_decomposition
    result = gelbach_decomposition(df, outcome="log_hourly_wage_real")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default block definitions for the Gelbach decomposition.
# "base" is the short regression (female + demographics).
# The remaining blocks define additional covariate groups whose
# contribution we want to measure in an order-invariant way.
DEFAULT_GELBACH_BLOCKS = {
    "base": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
    "geography": ["C(state_fips)"],
    "job_sorting": ["C(occupation_code)", "C(industry_code)", "C(class_of_worker)"],
    "schedule": ["usual_hours_week", "work_from_home", "commute_minutes_one_way"],
    "family": ["C(marital_status)", "number_children", "children_under_5"],
}

REPRODUCTIVE_GELBACH_BLOCKS = {
    "base": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
    "geography": ["C(state_fips)"],
    "job_sorting": ["C(occupation_code)", "C(industry_code)", "C(class_of_worker)"],
    "schedule": ["usual_hours_week", "work_from_home", "commute_minutes_one_way"],
    "family": ["C(marital_status)", "number_children", "children_under_5"],
    "reproductive": [
        "recent_birth", "recent_marriage", "has_own_child",
        "own_child_under6", "own_child_6_17_only",
        "C(couple_type)", "C(reproductive_stage)",
    ],
    "job_context": [
        "autonomy", "schedule_unpredictability", "time_pressure",
        "coordination_responsibility", "physical_proximity", "job_rigidity",
    ],
}


@dataclass
class GelbachResult:
    """Result of a Gelbach decomposition."""

    base_coef: float
    full_coef: float
    total_explained: float
    block_contributions: dict[str, float]
    block_ses: dict[str, float]
    n_obs: int
    r_squared_base: float
    r_squared_full: float
    identity_check: float  # should be ~0: base - full - sum(delta_k)


def gelbach_decomposition(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    treatment: str = "female",
    weight_col: str = "person_weight",
    blocks: dict[str, list[str]] | None = None,
    max_rows: int = 2_000_000,
) -> GelbachResult:
    """Run the Gelbach exact decomposition.

    Parameters
    ----------
    df : DataFrame
        Analysis-ready panel with outcome, treatment, weights, and all
        covariates referenced in *blocks*.
    outcome : str
        Dependent variable.
    treatment : str
        Variable of interest (default: "female").
    weight_col : str
        Survey weight column.
    blocks : dict
        Keys are block names; values are lists of column references
        (may include ``C(...)`` for categoricals).  The "base" block
        contains the short-regression controls.  All other blocks are
        the additional covariates whose contributions are decomposed.
    max_rows : int
        Safety guard against OOM.  Raise if the cleaned sample exceeds
        this.  Set to 0 to disable.  Default 2M rows — for the full
        9.7M pooled ACS, subsample or run year-by-year.

    Returns
    -------
    GelbachResult
    """
    if blocks is None:
        blocks = DEFAULT_GELBACH_BLOCKS.copy()

    blocks = {k: list(v) for k, v in blocks.items()}

    if "base" not in blocks:
        raise ValueError("blocks must contain a 'base' key with the short-regression controls")

    base_vars = blocks.pop("base")

    # Ensure treatment is in the base specification
    if treatment not in base_vars:
        base_vars = [treatment] + base_vars

    # Full model = base + all additional blocks
    additional_vars: list[str] = []
    for block_vars in blocks.values():
        additional_vars.extend(block_vars)

    full_vars = base_vars + additional_vars

    # Separate categorical vs numeric variable names
    categorical_vars = set()
    numeric_vars = set()
    for v in full_vars:
        if v.startswith("C(") and v.endswith(")"):
            categorical_vars.add(v[2:-1])
        else:
            numeric_vars.add(v)

    # Build clean frame with all needed columns
    clean = df[[outcome, weight_col]].copy()
    clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce")
    clean[weight_col] = pd.to_numeric(clean[weight_col], errors="coerce")

    for col_name in categorical_vars | numeric_vars:
        if col_name in df.columns:
            clean[col_name] = df[col_name]

    # Validity mask: require finite outcome and positive weight
    valid = clean[outcome].notna() & clean[weight_col].notna() & (clean[weight_col] > 0)

    # For numeric (non-categorical) variables, coerce and require non-NaN
    for col_name in numeric_vars:
        if col_name in clean.columns:
            s = pd.to_numeric(clean[col_name], errors="coerce")
            clean[col_name] = s
            valid &= s.notna()

    # For categorical variables, require non-NaN but do NOT coerce to numeric
    for col_name in categorical_vars:
        if col_name in clean.columns:
            valid &= clean[col_name].notna()

    clean = clean.loc[valid].reset_index(drop=True)

    if len(clean) < 100:
        raise ValueError(f"Too few valid observations for Gelbach decomposition: {len(clean)}")

    if max_rows > 0 and len(clean) > max_rows:
        raise ValueError(
            f"Gelbach decomposition uses dense matrices and would OOM on {len(clean):,} rows "
            f"(limit: {max_rows:,}). Subsample, run year-by-year, or set max_rows=0 to override."
        )

    y = clean[outcome].to_numpy(dtype=float)
    w = clean[weight_col].to_numpy(dtype=float)

    X_base = _build_design(clean, base_vars)
    X_full = _build_design(clean, full_vars)

    # Step 1: Fit base and full models
    beta_base, se_base, r2_base = _wls_fit(y, X_base, w)
    beta_full, se_full, r2_full = _wls_fit(y, X_full, w)

    # Female coefficient in each model
    treat_idx_base = list(X_base.columns).index(treatment)
    treat_idx_full = list(X_full.columns).index(treatment)

    base_coef = beta_base[treat_idx_base]
    full_coef = beta_full[treat_idx_full]
    total_explained = base_coef - full_coef

    # Step 2: For each additional block, compute delta_k
    # delta_k = sum over j in block_k of [ gamma_j * beta_full_j ]
    # where gamma_j = auxiliary regression coefficient of X_j on treatment
    # (controlling for base covariates)

    # Equivalent and simpler formulation: delta_k = Pi_k @ beta_full_k
    # where Pi_k are the coefficients from regressing X_k on the base variables
    # and beta_full_k are the full-model coefficients on block k variables.

    block_contributions = {}
    block_ses = {}

    for block_name, block_vars in blocks.items():
        X_block = _build_design(clean, block_vars, add_constant=False)
        if X_block.shape[1] == 0:
            block_contributions[block_name] = 0.0
            block_ses[block_name] = 0.0
            continue

        # Get the full-model coefficients for this block's variables
        block_coefs = []
        for col in X_block.columns:
            if col in X_full.columns:
                idx = list(X_full.columns).index(col)
                block_coefs.append(beta_full[idx])
            else:
                block_coefs.append(0.0)
        block_coefs = np.array(block_coefs)

        # Auxiliary regressions: regress each block variable on the base
        # variables and extract the coefficient on treatment
        gammas = np.zeros(len(block_coefs))
        for j, col in enumerate(X_block.columns):
            xj = X_block[col].to_numpy(dtype=float)
            aux_beta, _, _ = _wls_fit(xj, X_base, w)
            gammas[j] = aux_beta[treat_idx_base]

        delta_k = float(np.sum(gammas * block_coefs))
        block_contributions[block_name] = delta_k

        # Approximate SE via delta method (simplified):
        # Var(delta_k) ≈ sum_j (gamma_j^2 * var(beta_full_j))
        block_var = 0.0
        for j, col in enumerate(X_block.columns):
            if col in X_full.columns:
                idx = list(X_full.columns).index(col)
                block_var += gammas[j] ** 2 * se_full[idx] ** 2
        block_ses[block_name] = float(np.sqrt(max(block_var, 0.0)))

    # Identity check: base - full should equal sum of deltas
    sum_deltas = sum(block_contributions.values())
    identity_residual = total_explained - sum_deltas

    if abs(identity_residual) > 0.001:
        logger.warning(
            "Gelbach identity check: residual = %.6f (base=%.4f, full=%.4f, sum_delta=%.4f)",
            identity_residual, base_coef, full_coef, sum_deltas,
        )

    return GelbachResult(
        base_coef=base_coef,
        full_coef=full_coef,
        total_explained=total_explained,
        block_contributions=block_contributions,
        block_ses=block_ses,
        n_obs=len(clean),
        r_squared_base=r2_base,
        r_squared_full=r2_full,
        identity_check=identity_residual,
    )


def gelbach_to_dataframe(result: GelbachResult) -> pd.DataFrame:
    """Convert a GelbachResult to a tidy DataFrame for reporting."""
    rows = []
    for block_name, delta in result.block_contributions.items():
        se = result.block_ses.get(block_name, float("nan"))
        pct_of_total = (delta / result.total_explained * 100) if result.total_explained != 0 else float("nan")
        rows.append({
            "block": block_name,
            "delta": delta,
            "se": se,
            "pct_of_explained": pct_of_total,
        })

    df = pd.DataFrame(rows)
    # Add summary row
    summary = pd.DataFrame([{
        "block": "TOTAL",
        "delta": result.total_explained,
        "se": float("nan"),
        "pct_of_explained": 100.0,
    }])
    df = pd.concat([df, summary], ignore_index=True)

    df.attrs["base_coef"] = result.base_coef
    df.attrs["full_coef"] = result.full_coef
    df.attrs["n_obs"] = result.n_obs
    df.attrs["identity_check"] = result.identity_check

    return df


def _build_design(
    df: pd.DataFrame,
    var_specs: list[str],
    add_constant: bool = True,
) -> pd.DataFrame:
    """Build a design matrix from variable specifications.

    Handles ``C(varname)`` syntax for categorical dummies.
    """
    frames: list[pd.Series | pd.DataFrame] = []

    for spec in var_specs:
        if spec.startswith("C(") and spec.endswith(")"):
            col_name = spec[2:-1]
            if col_name not in df.columns:
                logger.debug("Gelbach: skipping missing categorical %s", col_name)
                continue
            dummies = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True, dtype=float)
            frames.append(dummies)
        else:
            if spec not in df.columns:
                logger.debug("Gelbach: skipping missing variable %s", spec)
                continue
            frames.append(df[spec].astype(float))

    if not frames:
        return pd.DataFrame(index=df.index)

    X = pd.concat(frames, axis=1)

    if add_constant:
        X.insert(0, "const", 1.0)

    return X


def _wls_fit(
    y: np.ndarray,
    X: pd.DataFrame,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit weighted least squares, return (coefficients, standard_errors, r_squared)."""
    y_arr = np.asarray(y, dtype=float)
    X_arr = X.to_numpy(dtype=float)
    w_arr = np.asarray(w, dtype=float)

    sqrt_w = np.sqrt(w_arr)
    X_w = X_arr * sqrt_w[:, None]
    y_w = y_arr * sqrt_w

    beta, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)

    fitted = X_arr @ beta
    resid = y_arr - fitted
    sse = float(np.sum(w_arr * resid ** 2))

    n, k = X_arr.shape
    dof = max(n - k, 1)
    sigma2 = sse / dof

    xtwx = X_w.T @ X_w
    xtwx_inv = np.linalg.pinv(xtwx)
    cov = sigma2 * xtwx_inv
    se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))

    weight_sum = float(np.sum(w_arr))
    y_mean = float(np.sum(w_arr * y_arr) / weight_sum)
    sst = float(np.sum(w_arr * (y_arr - y_mean) ** 2))
    r_squared = 1.0 - (sse / sst) if sst > 0 else float("nan")

    return beta, se, r_squared
