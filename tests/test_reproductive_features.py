"""Tests for reproductive-burden feature builders."""

import numpy as np
import pandas as pd

from gender_gap.features.reproductive import (
    add_fertility_risk_features,
    add_repro_interactions,
    add_reproductive_features,
)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "female": [1, 1, 1, 0, 1, 1, 1, 1],
            "age": [28, 31, 36, 33, 46, 52, 29, 41],
            "age_sq": [28**2, 31**2, 36**2, 33**2, 46**2, 52**2, 29**2, 41**2],
            "education_level": ["bachelors", "masters", "hs_diploma", "bachelors", "bachelors", "masters", "hs_diploma", "bachelors"],
            "race_ethnicity": ["white_non_hispanic", "white_non_hispanic", "black", "white_non_hispanic", "black", "white_non_hispanic", "hispanic", "white_non_hispanic"],
            "state_fips": [6, 6, 36, 6, 36, 6, 48, 12],
            "marital_status": ["never_married", "married", "married", "married", "never_married", "never_married", "married", "married"],
            "person_weight": np.ones(8),
            "noc": [0, 0, 1, 0, 0, 0, 0, 0],
            "paoc": [0, 0, 1, 0, 0, 0, 0, 0],
            "fer": [0, 1, 0, 0, 0, 0, 0, 1],
            "marhm": [0, 3, 0, 0, 0, 0, 12, 6],
            "cplt": [0, 1, 1, 1, 0, 0, 2, 1],
            "partner": [0, 1, 1, 1, 0, 0, 1, 1],
            "hourly_wage_real": [25, 30, 22, 32, 28, 24, 26, 29],
            "annual_earnings_real": [50000, 62000, 43000, 64000, 48000, 45000, 52000, 58000],
            "usual_hours_week": [40, 40, 35, 40, 40, 35, 40, 42],
            "weeks_worked": [52, 52, 50, 52, 52, 52, 52, 52],
        }
    )


def test_add_reproductive_features_builds_seeded_columns():
    df = add_reproductive_features(_base_frame())

    assert df["recent_birth"].tolist()[1] == 1
    assert df["recent_marriage"].tolist()[1] == 1
    assert df["same_sex_couple_household"].tolist()[6] == 1
    assert df["reproductive_stage"].tolist()[2] == "mother_under6"
    assert df["older_low_fertility_placebo"].tolist()[4] == 1
    assert df["ftfy_indicator"].tolist()[0] == 1


def test_add_fertility_risk_features_scores_childless_women():
    df = add_reproductive_features(_base_frame())
    scored = add_fertility_risk_features(df)

    assert scored["fertility_risk_score"].notna().sum() >= 4
    assert set(scored["fertility_risk_quartile"].dropna().unique()).issubset({"Q1", "Q2", "Q3", "Q4"})
    assert scored.loc[scored["female"].eq(0), "fertility_risk_score"].notna().all()
    assert scored.loc[scored["female"].eq(0), "fertility_risk_quartile"].notna().all()


def test_add_repro_interactions_builds_selected_terms():
    df = add_reproductive_features(_base_frame())
    df["autonomy"] = np.linspace(10, 80, len(df))
    df["job_rigidity"] = np.linspace(5, 40, len(df))

    interacted = add_repro_interactions(df)

    assert "female_x_recent_birth" in interacted.columns
    assert "female_x_autonomy" in interacted.columns
    assert "female_x_own_child_under6_x_job_rigidity" in interacted.columns


def test_add_reproductive_features_handles_missing_child_columns():
    df = _base_frame().drop(columns=["noc", "paoc", "fer", "marhm", "cplt", "partner"])

    enriched = add_reproductive_features(df)

    assert enriched["noc"].eq(0).all()
    assert enriched["paoc"].eq(0).all()
    assert enriched["recent_birth"].eq(0).all()


def test_add_reproductive_features_handles_missing_work_columns():
    df = _base_frame().drop(columns=["weeks_worked", "usual_hours_week"])

    enriched = add_reproductive_features(df)

    assert enriched["ftfy_indicator"].eq(0).all()
