"""Tests for employment-selection robustness models."""

import numpy as np
import pandas as pd

from gender_gap.models.selection import run_selection_robustness


def _selection_df(seed: int = 7, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    female = rng.integers(0, 2, n)
    age = rng.integers(25, 55, n).astype(float)
    children = rng.choice([0, 1, 2, 3], size=n, p=[0.45, 0.25, 0.2, 0.1])
    young_children = ((children > 0) & (rng.random(n) < 0.35)).astype(int)
    education = rng.choice(
        ["hs_diploma", "some_college", "bachelors", "graduate"],
        size=n,
        p=[0.25, 0.30, 0.25, 0.20],
    )
    race = rng.choice(
        ["white_non_hispanic", "black_non_hispanic", "hispanic"],
        size=n,
        p=[0.6, 0.18, 0.22],
    )
    marital = rng.choice(
        ["married", "never_married", "divorced"],
        size=n,
        p=[0.52, 0.33, 0.15],
    )
    state = rng.choice([6, 12, 36, 48], size=n)

    linp = (
        1.1
        - 0.75 * female
        + 0.02 * (age - 40)
        - 0.22 * young_children
        + 0.15 * (education == "graduate")
    )
    p_emp = 1 / (1 + np.exp(-linp))
    employed = (rng.random(n) < p_emp).astype(int)

    annual_earnings = np.zeros(n)
    hourly_wage = np.full(n, np.nan)
    employed_idx = employed == 1
    log_earn = (
        10.7
        - 0.18 * female[employed_idx]
        + 0.015 * (age[employed_idx] - 40)
        + 0.12 * (education[employed_idx] == "graduate")
        - 0.05 * young_children[employed_idx]
        + rng.normal(0, 0.25, employed_idx.sum())
    )
    annual_earnings[employed_idx] = np.exp(log_earn)
    hours = rng.choice([30, 35, 40, 45], size=n, p=[0.15, 0.20, 0.45, 0.20]).astype(float)
    hourly_wage[employed_idx] = annual_earnings[employed_idx] / (50 * hours[employed_idx])

    return pd.DataFrame({
        "female": female,
        "age": age,
        "age_sq": age ** 2,
        "race_ethnicity": race,
        "education_level": education,
        "marital_status": marital,
        "number_children": children,
        "children_under_5": young_children,
        "state_fips": state,
        "employed": employed,
        "annual_earnings_real": annual_earnings,
        "hourly_wage_real": hourly_wage,
        "person_weight": rng.uniform(80, 140, n),
    })


def test_selection_robustness_runs_and_returns_expected_columns():
    result = run_selection_robustness(_selection_df())

    assert list(result["model"]) == ["S0", "S1", "S2"]
    assert "employment_female_prob_pp" in result.columns
    assert "combined_expected_earnings_gap_pct" in result.columns
    assert "ipw_worker_hourly_gap_pct" in result.columns


def test_selection_detects_negative_female_employment_and_earnings_effect():
    result = run_selection_robustness(_selection_df())
    s2 = result.loc[result["model"] == "S2"].iloc[0]

    assert s2["employment_female_prob_pp"] < 0
    assert s2["female_conditional_earnings_pct"] < 0
    assert s2["combined_expected_earnings_gap_pct"] > 0


def test_selection_ipw_gap_is_finite():
    result = run_selection_robustness(_selection_df())
    assert np.isfinite(result["ipw_worker_hourly_gap_pct"]).all()
