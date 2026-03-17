"""Tests for ACS standardization."""

import pandas as pd
import pytest

from gender_gap.standardize.acs_standardize import standardize_acs
from gender_gap.standardize.schema import PERSON_YEAR_CORE_COLUMNS


@pytest.fixture
def mock_acs_raw():
    """Minimal mock ACS PUMS data."""
    return pd.DataFrame(
        {
            "SERIALNO": ["H001", "H001", "H002"],
            "SPORDER": [1, 2, 1],
            "SEX": [1, 2, 2],
            "AGEP": [35, 32, 45],
            "SCHL": [21, 22, 16],
            "MAR": [1, 1, 3],
            "HISP": [1, 1, 2],
            "RAC1P": [1, 1, 6],
            "OCCP": ["1010", "2100", "3500"],
            "INDP": ["6190", "7270", "8680"],
            "COW": [1, 1, 6],
            "WKHP": [40, 35, 50],
            "WKWN": [52, 48, 50],
            "JWTRNS": [1, 12, 3],
            "JWMNP": [25, 0, 45],
            "ST": ["06", "06", "36"],
            "PUMA": ["03700", "03700", "03800"],
            "POWSP": ["06", "06", "36"],
            "POWPUMA": ["03701", "03700", "03801"],
            "NOC": [2, 1, 0],
            "PAOC": [1, 2, 4],
            "WAGP": [75000, 85000, 60000],
            "PERNP": [78000, 87000, 62000],
            "ADJINC": [1050000, 1050000, 1050000],
            "PWGTP": [50, 45, 60],
            "FER": [0, 1, 0],
            "MARHM": [0, 5, 0],
            "CPLT": [1, 1, 0],
            "PARTNER": [1, 1, 0],
            "RELSHIPP": [20, 21, 37],
        }
    )


def test_standardize_produces_correct_columns(mock_acs_raw):
    result = standardize_acs(mock_acs_raw, survey_year=2022)
    assert list(result.columns) == PERSON_YEAR_CORE_COLUMNS


def test_standardize_female_coding(mock_acs_raw):
    result = standardize_acs(mock_acs_raw, survey_year=2022)
    assert result["female"].tolist() == [0, 1, 1]


def test_standardize_work_from_home(mock_acs_raw):
    result = standardize_acs(mock_acs_raw, survey_year=2022)
    assert result["work_from_home"].tolist() == [0, 1, 0]


def test_standardize_race_hispanic_override(mock_acs_raw):
    result = standardize_acs(mock_acs_raw, survey_year=2022)
    # Third person has HISP=2, so should be hispanic regardless of RAC1P
    assert result["race_ethnicity"].iloc[2] == "hispanic"


def test_standardize_education_recode(mock_acs_raw):
    result = standardize_acs(mock_acs_raw, survey_year=2022)
    assert result["education_level"].iloc[0] == "bachelors"
    assert result["education_level"].iloc[1] == "masters"
    assert result["education_level"].iloc[2] == "hs_diploma"


def test_standardize_hourly_wage_positive(mock_acs_raw):
    result = standardize_acs(mock_acs_raw, survey_year=2022)
    # All workers have positive hours and wages, so hourly should be positive
    assert (result["hourly_wage_real"].dropna() > 0).all()


def test_standardize_self_employed(mock_acs_raw):
    result = standardize_acs(mock_acs_raw, survey_year=2022)
    # COW=6 is self-employed
    assert result["self_employed"].iloc[2] == 1
    assert result["self_employed"].iloc[0] == 0


def test_standardize_family_fields_from_person_summaries(mock_acs_raw):
    result = standardize_acs(mock_acs_raw, survey_year=2022)

    assert result["number_children"].tolist() == [2, 1, 0]
    assert result["children_under_5"].tolist() == [1, 0, 0]


def test_standardize_reproductive_extension_fields(mock_acs_raw):
    result = standardize_acs(mock_acs_raw, survey_year=2022)

    assert result["recent_birth"].tolist() == [0, 1, 0]
    assert result["recent_marriage"].tolist() == [0, 1, 0]
    assert result["has_own_child"].tolist() == [1, 1, 0]
    assert result["same_sex_couple_household"].tolist() == [0, 0, 0]
    assert result["opposite_sex_couple_household"].tolist() == [1, 1, 0]
    assert result["reproductive_stage"].tolist() == [
        "mother_under6",
        "recent_birth",
        "childless_unpartnered",
    ]


def test_standardize_optionally_keeps_replicate_weights(mock_acs_raw):
    raw = mock_acs_raw.copy()
    raw["PWGTP1"] = [55, 49, 64]
    raw["PWGTP80"] = [45, 41, 58]

    result = standardize_acs(
        raw,
        survey_year=2022,
        keep_replicate_weights=True,
    )

    assert "PWGTP1" in result.columns
    assert "PWGTP80" in result.columns
    assert result["PWGTP1"].tolist() == [55, 49, 64]


def test_standardize_handles_numeric_fields_stored_as_strings(mock_acs_raw):
    raw = mock_acs_raw.astype({
        "SEX": "string",
        "AGEP": "string",
        "SCHL": "string",
        "MAR": "string",
        "HISP": "string",
        "RAC1P": "string",
        "COW": "string",
        "JWTRNS": "string",
        "NOC": "string",
        "PAOC": "string",
    })

    result = standardize_acs(raw, survey_year=2022)

    assert result["female"].tolist() == [0, 1, 1]
    assert result["education_level"].tolist() == ["bachelors", "masters", "hs_diploma"]
    assert result["children_under_5"].tolist() == [1, 0, 0]


def test_standardize_uses_state_fallback_for_2023plus_style_api_extracts(mock_acs_raw):
    raw = mock_acs_raw.drop(columns=["ST"]).copy()
    raw["STATE"] = ["06", "06", "36"]

    result = standardize_acs(raw, survey_year=2024)

    assert result["state_fips"].tolist() == ["06", "06", "36"]
