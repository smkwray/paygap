"""Tests for ATUS standardization."""

import pandas as pd

from gender_gap.standardize.atus_standardize import (
    standardize_atus_ipums,
    standardize_atus_summary,
)
from gender_gap.standardize.schema import PERSON_DAY_TIMEUSE_COLUMNS


def _make_atus_summary_row(**overrides):
    defaults = {
        "TUCASEID": "20230101130001",
        "TUYEAR": 2023,
        "TUDIARYDATE": "2023-06-15",
        "TESEX": 2,
        "TELFS": 1,
        "t050101": 300,
        "t050102": 60,
        "t050103": 30,
        "t180501": 45,
        "t020101": 60,
        "t020102": 30,
        "t030101": 90,
        "t030102": 30,
        "t030201": 15,
        "TUFINLWGT": 5000.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestATUSSummary:
    def test_produces_correct_columns(self):
        df = _make_atus_summary_row()
        result = standardize_atus_summary(df)
        assert list(result.columns) == PERSON_DAY_TIMEUSE_COLUMNS

    def test_female_coding(self):
        female = standardize_atus_summary(_make_atus_summary_row(TESEX=2))
        male = standardize_atus_summary(_make_atus_summary_row(TESEX=1))
        assert female["female"].iloc[0] == 1
        assert male["female"].iloc[0] == 0

    def test_paid_work_sums(self):
        result = standardize_atus_summary(_make_atus_summary_row())
        # 300 + 60 + 30 = 390 from t050101 + t050102 + t050103
        assert result["minutes_paid_work_diary"].iloc[0] >= 300

    def test_childcare_positive(self):
        result = standardize_atus_summary(_make_atus_summary_row())
        # 90 + 30 = 120
        assert result["minutes_childcare"].iloc[0] >= 90

    def test_weight_column(self):
        result = standardize_atus_summary(_make_atus_summary_row(TUFINLWGT=5000.0))
        assert result["person_weight"].iloc[0] == 5000.0


class TestATUSIPUMS:
    def test_produces_correct_columns(self):
        df = pd.DataFrame([{
            "CASEID": "123456",
            "YEAR": 2023,
            "DATE": "2023-06-15",
            "SEX": 2,
            "EMPSTAT": 1,
            "BLS_WORK": 480,
            "BLS_HHACT": 120,
            "BLS_CAREHH": 60,
            "WT06": 5000.0,
        }])
        result = standardize_atus_ipums(df)
        assert list(result.columns) == PERSON_DAY_TIMEUSE_COLUMNS
        assert result["minutes_paid_work_diary"].iloc[0] == 480
