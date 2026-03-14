"""Tests for SIPP standardization."""

import pandas as pd

from gender_gap.standardize.schema import PERSON_MONTH_CORE_COLUMNS
from gender_gap.standardize.sipp_standardize import standardize_sipp


def test_standardize_sipp_produces_schema_columns():
    raw = pd.DataFrame(
        {
            "SSUID": ["H1", "H1", "H2"],
            "PNUM": [1, 2, 1],
            "YEAR": [2023, 2023, 2023],
            "MONTHCODE": [1, 1, 1],
            "ESEX": [1, 2, 2],
            "RMESR": [1, 6, 8],
            "TJBHRS1": [40, 20, 0],
            "TJB1_MSUM": [5200, 1800, 0],
            "TJB1_OCC": ["2010", "4120", "0000"],
            "TJB1_IND": ["6470", "8770", "0000"],
            "TFIPSST": [6, 6, 36],
            "WPFINWGT": [100, 80, 60],
        }
    )

    result = standardize_sipp(raw)
    assert list(result.columns) == PERSON_MONTH_CORE_COLUMNS


def test_standardize_sipp_employment_mapping():
    raw = pd.DataFrame(
        {
            "SSUID": ["H1", "H1", "H2"],
            "PNUM": [1, 2, 1],
            "YEAR": [2023, 2023, 2023],
            "MONTHCODE": [1, 1, 1],
            "ESEX": [1, 2, 2],
            "RMESR": [1, 6, 8],
            "TJBHRS1": [40, 20, 0],
            "TJB1_MSUM": [5200, 1800, 0],
        }
    )

    result = standardize_sipp(raw)
    assert result["employed"].tolist() == [1, 0, 0]
    assert result["labor_force_status"].tolist() == ["employed", "unemployed", "not_in_labor_force"]


def test_standardize_sipp_derives_weekly_and_hourly_real():
    raw = pd.DataFrame(
        {
            "SSUID": ["H1", "H2"],
            "PNUM": [1, 1],
            "YEAR": [2022, 2023],
            "MONTHCODE": [1, 1],
            "ESEX": [2, 1],
            "RMESR": [1, 1],
            "TJBHRS1": [40, 20],
            "TJB1_MSUM": [5200, 2600],
            "WPFINWGT": [100, 100],
        }
    )
    cpi = {2022: 292.655, 2023: 304.702, 2024: 312.332}

    result = standardize_sipp(raw, cpi_index=cpi, base_year=2024)

    assert result["weekly_earnings_real"].notna().all()
    assert result["hourly_wage_real"].notna().all()
    assert (result["hourly_wage_real"] > 0).all()


def test_standardize_sipp_marks_multiple_jobholders_from_job2_earnings():
    raw = pd.DataFrame(
        {
            "SSUID": ["H1", "H2"],
            "PNUM": [1, 1],
            "YEAR": [2023, 2023],
            "MONTHCODE": [1, 1],
            "ESEX": [2, 1],
            "RMESR": [1, 1],
            "TJBHRS1": [40, 40],
            "TJB1_MSUM": [5200, 4200],
            "TJB2_MSUM": [0, 600],
        }
    )

    result = standardize_sipp(raw)
    assert result["multiple_jobholder"].tolist() == [0, 1]


def test_standardize_sipp_prefers_direct_hourly_pay_when_present():
    raw = pd.DataFrame(
        {
            "SSUID": ["H1"],
            "PNUM": [1],
            "YEAR": [2023],
            "MONTHCODE": [1],
            "ESEX": [2],
            "RMESR": [1],
            "TJBHRS1": [40],
            "TJB1_MSUM": [5200],
            "TJB1_HRLYPAY": [35],
        }
    )

    result = standardize_sipp(raw)
    assert result["paid_hourly"].iloc[0] == 1
    assert result["hourly_wage_real"].iloc[0] == 35


def test_standardize_sipp_handles_real_public_use_aliases():
    raw = pd.DataFrame(
        {
            "SSUID": ["H1", "H1"],
            "PNUM": [1, 2],
            "MONTHCODE": [1, 1],
            "ESEX": [2, 1],
            "RMESR": [1, 1],
            "TJB1_JOBHRS1": [40, 35],
            "TJB1_HOURLY1": [25.0, pd.NA],
            "EJB1_TYPPAY1": [1, 2],
            "TPEARN": [4000.0, 3500.0],
            "TJB1_OCC": ["4120", ""],
            "TJB1_IND": ["8770", ""],
            "WPFINWGT": [100.0, 80.0],
        }
    )

    result = standardize_sipp(raw, survey_year=2023)

    assert result["calendar_year"].tolist() == [2023, 2023]
    assert result["usual_hours_week"].tolist() == [40, 35]
    assert result["actual_hours_last_week"].tolist() == [40, 35]
    assert result["paid_hourly"].tolist() == [1, 0]
    assert result["hourly_wage_real"].iloc[0] == 25.0
    assert result["hourly_wage_real"].iloc[1] > 0
    assert result["occupation_code"].iloc[0] == "4120"
    assert result["industry_code"].iloc[0] == "8770"
    assert pd.isna(result["occupation_code"].iloc[1])
    assert pd.isna(result["industry_code"].iloc[1])
