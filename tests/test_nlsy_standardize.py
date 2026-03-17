"""Tests for NLSY standardization and g-proxy computation."""

import numpy as np
import pandas as pd
import pytest

from gender_gap.standardize import nlsy_standardize
from gender_gap.standardize.nlsy_standardize import (
    _recode_nlsy_education,
    _recode_nlsy_race,
    _resolve_nlsy_dir,
    compute_g_proxy,
    standardize_nlsy79_for_gap,
)


def _make_nlsy79_row(**overrides):
    """Create a single-row NLSY79-like DataFrame."""
    np.random.seed(42)
    defaults = {
        "person_id": 1001,
        "sex": 2,
        "age": 41,
        "age_2000": 41,
        "birth_year": 1959,
        "race_ethnicity_3cat": "NON-BLACK, NON-HISPANIC",
        "education_years": 16.0,
        "marital_status_2000": 1,
        "num_children_2000": 2,
        "occupation_code_2000": 110.0,
        "annual_earnings": 45000.0,
        "household_income": 85000.0,
        "net_worth": 200000.0,
        "parent_education": 12.0,
        "sample_weight_96": 500000,
        # ASVAB subtests (age-residualized raw scores)
        "GS": 500.0,
        "AR": 600.0,
        "WK": 700.0,
        "PC": 550.0,
        "NO": 800.0,
        "CS": 1200.0,
        "AS": 100.0,
        "MK": 500.0,
        "MC": 400.0,
        "EI": 500.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestGProxy:
    def test_compute_g_proxy_returns_series(self):
        df = _make_nlsy79_row()
        g = compute_g_proxy(df)
        assert isinstance(g, pd.Series)
        assert len(g) == 1

    def test_g_proxy_is_standardized(self):
        """With multiple rows, g_proxy should be z-scored."""
        np.random.seed(42)
        rows = []
        for i in range(100):
            row = {
                "GS": np.random.normal(500, 100),
                "AR": np.random.normal(600, 100),
                "WK": np.random.normal(700, 100),
                "PC": np.random.normal(550, 100),
                "NO": np.random.normal(800, 100),
                "CS": np.random.normal(1200, 100),
                "AS": np.random.normal(100, 100),
                "MK": np.random.normal(500, 100),
                "MC": np.random.normal(400, 100),
                "EI": np.random.normal(500, 100),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        g = compute_g_proxy(df)
        assert abs(g.mean()) < 0.01
        assert abs(g.std() - 1.0) < 0.05

    def test_g_proxy_missing_subtests_still_computes(self):
        """With only some subtests, g_proxy still computes from available ones."""
        df = pd.DataFrame({"GS": [500, 600], "AR": [600, 700]})
        g = compute_g_proxy(df)
        assert len(g) == 2
        assert g.notna().all()

    def test_g_proxy_no_subtests_raises(self):
        df = pd.DataFrame({"unrelated": [1]})
        with pytest.raises(ValueError):
            compute_g_proxy(df)


class TestStandardizeNLSY79:
    def test_female_coding(self):
        female = standardize_nlsy79_for_gap(_make_nlsy79_row(sex=2))
        male = standardize_nlsy79_for_gap(_make_nlsy79_row(sex=1))
        assert female["female"].iloc[0] == 1
        assert male["female"].iloc[0] == 0

    def test_has_g_proxy(self):
        # Need multiple rows for z-scoring to produce non-NaN values
        rows = pd.concat(
            [_make_nlsy79_row(person_id=i, GS=500 + i * 10) for i in range(5)],
            ignore_index=True,
        )
        result = standardize_nlsy79_for_gap(rows)
        assert "g_proxy" in result.columns
        assert result["g_proxy"].notna().all()

    def test_has_earnings(self):
        result = standardize_nlsy79_for_gap(
            _make_nlsy79_row(annual_earnings=45000.0)
        )
        assert result["annual_earnings_real"].iloc[0] > 0

    def test_race_recode(self):
        assert _recode_nlsy_race(
            pd.Series(["NON-BLACK, NON-HISPANIC"])
        ).iloc[0] == "white_non_hispanic"
        assert _recode_nlsy_race(pd.Series(["BLACK"])).iloc[0] == "black"
        assert _recode_nlsy_race(pd.Series(["HISPANIC"])).iloc[0] == "hispanic"

    def test_education_recode(self):
        assert _recode_nlsy_education(pd.Series([10.0])).iloc[0] == "less_than_hs"
        assert _recode_nlsy_education(pd.Series([12.0])).iloc[0] == "hs_diploma"
        assert _recode_nlsy_education(pd.Series([16.0])).iloc[0] == "bachelors"
        assert _recode_nlsy_education(pd.Series([18.0])).iloc[0] == "masters"

    def test_data_source_label(self):
        result = standardize_nlsy79_for_gap(_make_nlsy79_row())
        assert result["data_source"].iloc[0] == "NLSY79"


def test_resolve_nlsy_dir_prefers_local_then_shared(tmp_path, monkeypatch):
    local_dir = tmp_path / "paygap" / "data" / "external" / "nlsy"
    shared_dir = tmp_path / "shared" / "sources" / "misc" / "large_payloads" / "wave4" / "paygap" / "processed" / "nlsy"
    local_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    monkeypatch.setattr(nlsy_standardize, "DEFAULT_LOCAL_NLSY_DIR", local_dir)
    monkeypatch.setattr(nlsy_standardize, "DEFAULT_SHARED_NLSY_DIR", shared_dir)
    monkeypatch.delenv("NLSY_DATA_DIR", raising=False)

    assert _resolve_nlsy_dir() == local_dir

    local_dir.rmdir()
    assert _resolve_nlsy_dir() == shared_dir
