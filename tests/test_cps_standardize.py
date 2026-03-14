"""Tests for CPS ORG standardization."""

import pandas as pd
import pytest

from gender_gap.standardize.cps_standardize import (
    standardize_cps_ipums,
    standardize_cps_official,
)
from gender_gap.standardize.schema import PERSON_MONTH_CORE_COLUMNS


def _make_ipums_cps_row(**overrides):
    """Create a single-row IPUMS CPS DataFrame."""
    defaults = {
        "YEAR": 2023,
        "MONTH": 6,
        "SERIAL": 100,
        "PERNUM": 1,
        "SEX": 2,
        "EMPSTAT": 10,
        "UHRSWORKORG": 40,
        "AHRSWORKT": 42,
        "PAIDHOUR": 2,
        "HOURWAGE": 25.0,
        "EARNWEEK": 1000.0,
        "OTPAY": 2,
        "MULTJOB": 1,
        "OCC": 1010,
        "IND": 7860,
        "STATEFIP": 6,
        "EARNWT": 1500.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestCPSIPUMS:
    def test_produces_correct_columns(self):
        df = _make_ipums_cps_row()
        result = standardize_cps_ipums(df)
        assert list(result.columns) == PERSON_MONTH_CORE_COLUMNS

    def test_female_coding(self):
        female = standardize_cps_ipums(_make_ipums_cps_row(SEX=2))
        male = standardize_cps_ipums(_make_ipums_cps_row(SEX=1))
        assert female["female"].iloc[0] == 1
        assert male["female"].iloc[0] == 0

    def test_employment_status(self):
        employed = standardize_cps_ipums(_make_ipums_cps_row(EMPSTAT=10))
        unemployed = standardize_cps_ipums(_make_ipums_cps_row(EMPSTAT=21))
        nilf = standardize_cps_ipums(_make_ipums_cps_row(EMPSTAT=36))
        assert employed["employed"].iloc[0] == 1
        assert unemployed["employed"].iloc[0] == 0
        assert nilf["employed"].iloc[0] == 0

    def test_labor_force_status(self):
        result = standardize_cps_ipums(_make_ipums_cps_row(EMPSTAT=21))
        assert result["labor_force_status"].iloc[0] == "unemployed"

    def test_hourly_wage_direct(self):
        result = standardize_cps_ipums(_make_ipums_cps_row(HOURWAGE=25.0))
        assert result["hourly_wage_real"].iloc[0] == pytest.approx(25.0)

    def test_hourly_wage_derived_from_weekly(self):
        result = standardize_cps_ipums(
            _make_ipums_cps_row(HOURWAGE=0.0, EARNWEEK=1000.0, UHRSWORKORG=40)
        )
        assert result["hourly_wage_real"].iloc[0] == pytest.approx(25.0)

    def test_overtime_indicator(self):
        has_ot = standardize_cps_ipums(_make_ipums_cps_row(OTPAY=2))
        no_ot = standardize_cps_ipums(_make_ipums_cps_row(OTPAY=1))
        assert has_ot["overtime_indicator"].iloc[0] == 1
        assert no_ot["overtime_indicator"].iloc[0] == 0

    def test_cpi_deflation(self):
        cpi = {2023: 304.7, 2024: 314.0}
        result = standardize_cps_ipums(
            _make_ipums_cps_row(HOURWAGE=25.0, YEAR=2023),
            cpi_index=cpi,
            base_year=2024,
        )
        expected = 25.0 * (314.0 / 304.7)
        assert result["hourly_wage_real"].iloc[0] == pytest.approx(expected, rel=1e-3)

    def test_weight_uses_earnwt(self):
        result = standardize_cps_ipums(_make_ipums_cps_row(EARNWT=1500.0))
        assert result["person_weight"].iloc[0] == 1500.0


class TestCPSOfficial:
    def _make_official_row(self, **overrides):
        defaults = {
            "HRYEAR4": 2023,
            "HRMONTH": 6,
            "HRHHID": "ABC123",
            "PULINENO": 1,
            "PESEX": 2,
            "PEMLR": 1,
            "PEHRUSL1": 40,
            "PEHRACT1": 42,
            "PEERNHRO": 1,
            "PTERNHLY": 2500,  # cents
            "PTERNWA": 100000,  # cents
            "PEERNUOT": 1,
            "PEMJOT": 2,
            "PRMJOCC1": 100,
            "PRMJIND1": 200,
            "GESTFIPS": 6,
            "PWSSWGT": 1200.0,
        }
        defaults.update(overrides)
        return pd.DataFrame([defaults])

    def test_produces_correct_columns(self):
        result = standardize_cps_official(self._make_official_row())
        assert list(result.columns) == PERSON_MONTH_CORE_COLUMNS

    def test_female_coding(self):
        result = standardize_cps_official(self._make_official_row(PESEX=2))
        assert result["female"].iloc[0] == 1

    def test_hourly_wage_from_cents(self):
        result = standardize_cps_official(self._make_official_row(PTERNHLY=2500))
        assert result["hourly_wage_real"].iloc[0] == pytest.approx(25.0)
