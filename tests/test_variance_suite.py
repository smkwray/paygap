"""Tests for variance and tails diagnostics."""

import numpy as np
import pandas as pd
from gender_gap.models.variance_suite import run_variance_suite

from gender_gap.models import ols as ols_module


def test_run_variance_suite_returns_seeded_metrics():
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame(
        {
            "female": np.r_[np.zeros(n // 2), np.ones(n // 2)],
            "age": rng.randint(25, 55, n),
            "race_ethnicity": ["white_non_hispanic"] * n,
            "education_level": ["bachelors"] * n,
            "state_fips": [6] * n,
            "occupation_code": [1010] * n,
            "industry_code": [7860] * n,
            "class_of_worker": [1] * n,
            "usual_hours_week": rng.choice([35, 40, 45], n),
            "work_from_home": rng.binomial(1, 0.2, n),
            "commute_minutes_one_way": rng.uniform(5, 60, n),
            "marital_status": ["married"] * n,
            "number_children": rng.choice([0, 1, 2], n),
            "children_under_5": rng.binomial(1, 0.3, n),
            "person_weight": np.ones(n),
            "log_hourly_wage_real": rng.normal(3.2, 0.3, n),
            "reproductive_stage": rng.choice(["childless_unpartnered", "mother_under6"], n),
            "fertility_risk_quartile": rng.choice(["Q1", "Q2", "Q3", "Q4"], n),
            "job_rigidity_quartile": rng.choice(["Q1", "Q2", "Q3", "Q4"], n),
            "couple_type": rng.choice(["unpartnered", "opposite_sex"], n),
        }
    )
    df["age_sq"] = df["age"] ** 2

    result = run_variance_suite(df)
    assert not result.empty
    assert {
        "raw_variance_ratio",
        "residual_variance_ratio",
        "male_top10_share",
        "female_top10_share",
    }.issubset(set(result["metric"]))


def test_run_variance_suite_with_sparse_design(monkeypatch):
    monkeypatch.setattr(ols_module, "SPARSE_DESIGN_ROW_THRESHOLD", 10)
    rng = np.random.RandomState(1)
    n = 200
    df = pd.DataFrame(
        {
            "female": np.r_[np.zeros(n // 2), np.ones(n // 2)],
            "age": rng.randint(25, 55, n),
            "race_ethnicity": ["white_non_hispanic"] * n,
            "education_level": ["bachelors"] * n,
            "state_fips": [6] * n,
            "occupation_code": [1010] * n,
            "industry_code": [7860] * n,
            "class_of_worker": [1] * n,
            "usual_hours_week": rng.choice([35, 40, 45], n),
            "work_from_home": rng.binomial(1, 0.2, n),
            "commute_minutes_one_way": rng.uniform(5, 60, n),
            "marital_status": ["married"] * n,
            "number_children": rng.choice([0, 1, 2], n),
            "children_under_5": rng.binomial(1, 0.3, n),
            "person_weight": np.ones(n),
            "log_hourly_wage_real": rng.normal(3.1, 0.25, n),
            "reproductive_stage": rng.choice(["childless_unpartnered", "mother_under6"], n),
            "fertility_risk_quartile": rng.choice(["Q1", "Q2", "Q3", "Q4"], n),
            "job_rigidity_quartile": rng.choice(["Q1", "Q2", "Q3", "Q4"], n),
            "couple_type": rng.choice(["unpartnered", "opposite_sex"], n),
        }
    )
    df["age_sq"] = df["age"] ** 2

    result = run_variance_suite(df)

    assert not result.empty


def test_run_variance_suite_auto_builds_log_outcome():
    rng = np.random.RandomState(2)
    n = 120
    df = pd.DataFrame(
        {
            "female": np.r_[np.zeros(n // 2), np.ones(n // 2)],
            "age": rng.randint(25, 55, n),
            "race_ethnicity": ["white_non_hispanic"] * n,
            "education_level": ["bachelors"] * n,
            "state_fips": [6] * n,
            "occupation_code": [1010] * n,
            "industry_code": [7860] * n,
            "class_of_worker": [1] * n,
            "usual_hours_week": rng.choice([35, 40, 45], n),
            "work_from_home": rng.binomial(1, 0.2, n),
            "commute_minutes_one_way": rng.uniform(5, 60, n),
            "marital_status": ["married"] * n,
            "number_children": rng.choice([0, 1, 2], n),
            "children_under_5": rng.binomial(1, 0.3, n),
            "person_weight": np.ones(n),
            "hourly_wage_real": rng.uniform(15, 50, n),
            "reproductive_stage": rng.choice(["childless_unpartnered", "mother_under6"], n),
            "fertility_risk_quartile": rng.choice(["Q1", "Q2", "Q3", "Q4"], n),
            "job_rigidity_quartile": rng.choice(["Q1", "Q2", "Q3", "Q4"], n),
            "couple_type": rng.choice(["unpartnered", "opposite_sex"], n),
        }
    )
    df["age_sq"] = df["age"] ** 2

    result = run_variance_suite(df)

    assert not result.empty
    assert "metric" in result.columns


def test_run_variance_suite_accepts_occupation_stratifier():
    rng = np.random.RandomState(3)
    n = 160
    df = pd.DataFrame(
        {
            "female": np.r_[np.zeros(n // 2), np.ones(n // 2)],
            "age": rng.randint(25, 55, n),
            "race_ethnicity": ["white_non_hispanic"] * n,
            "education_level": ["bachelors"] * n,
            "state_fips": [6] * n,
            "occupation_code": rng.choice([1010, 2330, 4120, 5110], n),
            "industry_code": [7860] * n,
            "class_of_worker": [1] * n,
            "usual_hours_week": rng.choice([35, 40, 45], n),
            "work_from_home": rng.binomial(1, 0.2, n),
            "commute_minutes_one_way": rng.uniform(5, 60, n),
            "marital_status": ["married"] * n,
            "number_children": rng.choice([0, 1, 2], n),
            "children_under_5": rng.binomial(1, 0.2, n),
            "person_weight": np.ones(n),
            "log_hourly_wage_real": rng.normal(3.2, 0.3, n),
        }
    )
    df["age_sq"] = df["age"] ** 2

    result = run_variance_suite(df, stratifiers=["occupation_code"])

    occupation_rows = result.loc[result["stratifier"] == "occupation_code"]
    assert not occupation_rows.empty
    assert set(occupation_rows["stratum"].astype(str)) == {"1010", "2330", "4120", "5110"}
