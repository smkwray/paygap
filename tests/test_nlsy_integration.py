"""Integration test: run NLSY standardization on real data.

Requires the NLSY processed data to exist at NLSY_DATA_DIR.
Skips gracefully if data is not available.
"""

import os
from pathlib import Path

import numpy as np
import pytest

NLSY_DIR = Path(os.environ.get("NLSY_DATA_DIR", "data/external/nlsy"))

NLSY79_PATH = NLSY_DIR / "nlsy79_cfa_resid.csv"
NLSY97_PATH = NLSY_DIR / "nlsy97_cfa_resid.csv"

has_nlsy79 = NLSY79_PATH.exists()
has_nlsy97 = NLSY97_PATH.exists()


@pytest.mark.skipif(not has_nlsy79, reason="NLSY79 data not available")
class TestNLSY79Integration:
    @pytest.fixture(scope="class")
    def nlsy79_standardized(self):
        from gender_gap.standardize.nlsy_standardize import (
            load_nlsy79,
            standardize_nlsy79_for_gap,
        )
        df = load_nlsy79()
        return standardize_nlsy79_for_gap(df)

    def test_has_rows(self, nlsy79_standardized):
        assert len(nlsy79_standardized) > 100

    def test_has_required_columns(self, nlsy79_standardized):
        required = ["person_id", "female", "age", "g_proxy",
                     "annual_earnings_real", "education_years"]
        for col in required:
            assert col in nlsy79_standardized.columns, f"Missing: {col}"

    def test_g_proxy_is_standardized(self, nlsy79_standardized):
        g = nlsy79_standardized["g_proxy"].dropna()
        assert len(g) > 50
        # z-scored: mean ≈ 0, std ≈ 1
        assert abs(g.mean()) < 0.15
        assert 0.7 < g.std() < 1.3

    def test_female_is_binary(self, nlsy79_standardized):
        assert set(nlsy79_standardized["female"].dropna().unique()).issubset({0, 1})

    def test_can_run_ols_with_g_proxy(self, nlsy79_standardized):
        from gender_gap.models.ols import NLSY_BLOCK_DEFINITIONS, run_sequential_ols
        df = nlsy79_standardized.copy()
        # Need log wage
        df["hourly_wage_real"] = df["annual_earnings_real"] / (40 * 50)
        df["log_hourly_wage_real"] = np.log(
            df["hourly_wage_real"].replace(0, np.nan).clip(lower=0.01)
        )
        df["age_sq"] = df["age"] ** 2
        df["person_weight"] = 1.0  # equal weight for test

        # Filter to valid obs
        valid = (
            df["log_hourly_wage_real"].notna() &
            df["female"].notna() &
            df["age"].notna() &
            df["annual_earnings_real"].gt(0)
        )
        df = df[valid]

        if len(df) < 100:
            pytest.skip("Too few valid observations")

        results = run_sequential_ols(
            df,
            blocks=NLSY_BLOCK_DEFINITIONS,
            weight_col="person_weight",
        )
        assert len(results) > 0
        # The g_proxy model (N2) should exist if g_proxy has variance
        model_names = [r.model_name for r in results]
        assert "N0" in model_names
        # Female coefficient should be negative (women earn less)
        n0 = [r for r in results if r.model_name == "N0"][0]
        assert n0.female_coef < 0


@pytest.mark.skipif(not has_nlsy97, reason="NLSY97 data not available")
class TestNLSY97Integration:
    @pytest.fixture(scope="class")
    def nlsy97_standardized(self):
        from gender_gap.standardize.nlsy_standardize import (
            load_nlsy97,
            standardize_nlsy97_for_gap,
        )
        df = load_nlsy97()
        return standardize_nlsy97_for_gap(df)

    def test_has_rows(self, nlsy97_standardized):
        assert len(nlsy97_standardized) > 100

    def test_has_g_proxy(self, nlsy97_standardized):
        assert "g_proxy" in nlsy97_standardized.columns
        g = nlsy97_standardized["g_proxy"].dropna()
        assert len(g) > 50

    def test_has_female(self, nlsy97_standardized):
        assert "female" in nlsy97_standardized.columns
        assert set(nlsy97_standardized["female"].dropna().unique()).issubset({0, 1})

    def test_earnings_positive_subset(self, nlsy97_standardized):
        earners = nlsy97_standardized[
            nlsy97_standardized["annual_earnings_real"].gt(0)
        ]
        assert len(earners) > 50
