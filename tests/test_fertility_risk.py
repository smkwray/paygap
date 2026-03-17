"""Tests for fertility-risk model outputs."""

import numpy as np
import pandas as pd

from gender_gap.features.reproductive import add_fertility_risk_features, add_reproductive_features
from gender_gap.models.fertility_risk import run_fertility_risk_penalty


def test_run_fertility_risk_penalty_emits_quartile_summary_for_women_only_sample():
    main_n = 24
    placebo_n = 8
    ages = np.concatenate([np.linspace(25, 44, main_n), np.linspace(45, 52, placebo_n)]).astype(int)
    recent_birth = np.array(([0, 1] * 12) + ([0, 1] * 4))
    df = pd.DataFrame(
        {
            "female": np.ones(main_n + placebo_n, dtype=int),
            "age": ages,
            "age_sq": ages**2,
            "education_level": ["bachelors", "masters", "hs_diploma", "some_college"] * 8,
            "race_ethnicity": ["white_non_hispanic", "black", "hispanic", "asian"] * 8,
            "state_fips": [6, 12, 36, 48] * 8,
            "marital_status": ["never_married", "married", "separated", "married"] * 8,
            "person_weight": np.linspace(1.0, 2.5, main_n + placebo_n),
            "noc": np.zeros(main_n + placebo_n, dtype=int),
            "paoc": np.zeros(main_n + placebo_n, dtype=int),
            "fer": recent_birth,
            "marhm": ([0, 4, 0, 7] * 8)[: main_n + placebo_n],
            "cplt": ([0, 1, 3, 1] * 8)[: main_n + placebo_n],
            "partner": ([0, 1, 1, 1] * 8)[: main_n + placebo_n],
            "hourly_wage_real": np.linspace(18.0, 42.0, main_n + placebo_n),
            "annual_earnings_real": np.linspace(32000.0, 91000.0, main_n + placebo_n),
            "usual_hours_week": np.linspace(32.0, 48.0, main_n + placebo_n),
            "weeks_worked": np.repeat(52, main_n + placebo_n),
            "employment_indicator": np.ones(main_n + placebo_n, dtype=int),
            "ftfy_indicator": np.ones(main_n + placebo_n, dtype=int),
            "log_hourly_wage_real": np.log(np.linspace(18.0, 42.0, main_n + placebo_n)),
            "log_annual_earnings_real": np.log(np.linspace(32000.0, 91000.0, main_n + placebo_n)),
        }
    )

    featured = add_fertility_risk_features(add_reproductive_features(df))
    penalty, quartiles = run_fertility_risk_penalty(featured)

    assert not penalty.empty
    assert not quartiles.empty
    assert set(quartiles["sample"]) >= {"childless_25_44", "childless_45_49", "childless_50_54"}
    assert set(quartiles["risk_quartile"]).issubset({"Q1", "Q2", "Q3", "Q4"})
    assert quartiles["mean_outcome"].notna().all()
