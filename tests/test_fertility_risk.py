"""Tests for fertility-risk model outputs."""

import numpy as np
import pandas as pd

from gender_gap.features.reproductive import add_fertility_risk_features, add_reproductive_features
from gender_gap.models.fertility_risk import run_fertility_risk_penalty


def test_run_fertility_risk_penalty_emits_quartile_summary_for_women_only_sample():
    main_n = 24
    men_n = 24
    placebo_n = 8
    women_ages = np.linspace(25, 44, main_n).astype(int)
    men_ages = np.linspace(25, 44, men_n).astype(int)
    placebo_ages = np.linspace(45, 52, placebo_n).astype(int)
    ages = np.concatenate([women_ages, men_ages, placebo_ages])
    recent_birth = np.array(([0, 1] * 12) + ([0] * men_n) + ([0, 1] * 4))
    total_n = main_n + men_n + placebo_n
    df = pd.DataFrame(
        {
            "female": np.concatenate(
                [
                    np.ones(main_n, dtype=int),
                    np.zeros(men_n, dtype=int),
                    np.ones(placebo_n, dtype=int),
                ]
            ),
            "age": ages,
            "age_sq": ages**2,
            "education_level": (["bachelors", "masters", "hs_diploma", "some_college"] * 14)[:total_n],
            "race_ethnicity": (["white_non_hispanic", "black", "hispanic", "asian"] * 14)[:total_n],
            "state_fips": ([6, 12, 36, 48] * 14)[:total_n],
            "marital_status": (["never_married", "married", "separated", "married"] * 14)[:total_n],
            "person_weight": np.linspace(1.0, 2.5, total_n),
            "noc": np.zeros(total_n, dtype=int),
            "paoc": np.zeros(total_n, dtype=int),
            "fer": recent_birth,
            "marhm": ([0, 4, 0, 7] * 14)[:total_n],
            "cplt": ([0, 1, 3, 1] * 14)[:total_n],
            "partner": ([0, 1, 1, 1] * 14)[:total_n],
            "hourly_wage_real": np.linspace(18.0, 42.0, total_n),
            "annual_earnings_real": np.linspace(32000.0, 91000.0, total_n),
            "usual_hours_week": np.linspace(32.0, 48.0, total_n),
            "weeks_worked": np.repeat(52, total_n),
            "employment_indicator": np.ones(total_n, dtype=int),
            "ftfy_indicator": np.ones(total_n, dtype=int),
            "log_hourly_wage_real": np.log(np.linspace(18.0, 42.0, total_n)),
            "log_annual_earnings_real": np.log(np.linspace(32000.0, 91000.0, total_n)),
        }
    )

    featured = add_fertility_risk_features(add_reproductive_features(df))
    penalty, quartiles = run_fertility_risk_penalty(featured)

    assert not penalty.empty
    assert not quartiles.empty
    assert set(quartiles["sample"]) >= {
        "childless_25_44",
        "childless_men_25_44",
        "childless_45_49",
        "childless_50_54",
    }
    assert "childless_men_25_44" in set(penalty["sample"])
    assert set(quartiles["risk_quartile"]).issubset({"Q1", "Q2", "Q3", "Q4"})
    assert quartiles["mean_outcome"].notna().all()
