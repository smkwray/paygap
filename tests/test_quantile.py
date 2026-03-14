"""Tests for quantile regression models."""

import numpy as np
import pandas as pd
import pytest

from gender_gap.models.quantile import (
    QuantileResult,
    diagnose_distributional_pattern,
    quantile_results_to_dataframe,
    run_quantile_regression,
)


@pytest.fixture
def quantile_df():
    """DataFrame for quantile regression tests."""
    rng = np.random.RandomState(42)
    n = 600
    female = rng.binomial(1, 0.5, n)
    age = rng.uniform(25, 55, n)
    age_sq = age ** 2
    noise = rng.normal(0, 0.2, n)

    # Create wages where the gap widens at the top (glass ceiling)
    base = 2.5 + 0.02 * age - 0.0002 * age_sq
    gap = -0.10 - 0.15 * (base - base.mean()) / base.std()  # gap widens with higher base
    log_wage = base + gap * female + noise

    return pd.DataFrame({
        "female": female,
        "age": age,
        "age_sq": age_sq,
        "log_hourly_wage_real": log_wage,
        "person_weight": np.ones(n),
    })


class TestRunQuantileRegression:
    def test_default_quantiles(self, quantile_df):
        results = run_quantile_regression(quantile_df)
        assert len(results) == 5
        quantiles = [r.quantile for r in results]
        assert quantiles == [0.10, 0.25, 0.50, 0.75, 0.90]

    def test_custom_quantiles(self, quantile_df):
        results = run_quantile_regression(
            quantile_df, quantiles=[0.25, 0.50, 0.75]
        )
        assert len(results) == 3

    def test_result_attributes(self, quantile_df):
        results = run_quantile_regression(quantile_df)
        for r in results:
            assert isinstance(r, QuantileResult)
            assert 0 < r.quantile < 1
            assert r.female_se > 0
            assert 0 <= r.female_pvalue <= 1
            assert r.n_obs > 0

    def test_female_coef_negative(self, quantile_df):
        results = run_quantile_regression(quantile_df)
        for r in results:
            assert r.female_coef < 0, f"Q{r.quantile} should have negative female coef"

    def test_too_few_observations(self, quantile_df):
        small_df = quantile_df.iloc[:10]
        with pytest.raises(ValueError, match="Too few observations"):
            run_quantile_regression(small_df)

    def test_auto_log_wage(self, quantile_df):
        df = quantile_df.drop(columns=["log_hourly_wage_real"]).copy()
        df["hourly_wage_real"] = np.exp(quantile_df["log_hourly_wage_real"])
        results = run_quantile_regression(df)
        assert len(results) == 5

    def test_custom_controls(self, quantile_df):
        results = run_quantile_regression(
            quantile_df, controls=["age"]
        )
        assert len(results) == 5


class TestQuantileResultsToDataframe:
    def test_conversion(self, quantile_df):
        results = run_quantile_regression(quantile_df)
        df = quantile_results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        expected_cols = {"quantile", "female_coef", "female_se", "female_pvalue", "n_obs"}
        assert expected_cols == set(df.columns)

    def test_empty_results(self):
        df = quantile_results_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestDiagnoseDistributionalPattern:
    def test_glass_ceiling(self):
        results = [
            QuantileResult(0.10, -0.05, 0.01, 0.001, 500),
            QuantileResult(0.50, -0.10, 0.01, 0.001, 500),
            QuantileResult(0.90, -0.20, 0.01, 0.001, 500),
        ]
        assert diagnose_distributional_pattern(results) == "glass_ceiling"

    def test_sticky_floor(self):
        results = [
            QuantileResult(0.10, -0.20, 0.01, 0.001, 500),
            QuantileResult(0.50, -0.10, 0.01, 0.001, 500),
            QuantileResult(0.90, -0.05, 0.01, 0.001, 500),
        ]
        assert diagnose_distributional_pattern(results) == "sticky_floor"

    def test_uniform(self):
        results = [
            QuantileResult(0.10, -0.10, 0.01, 0.001, 500),
            QuantileResult(0.50, -0.11, 0.01, 0.001, 500),
            QuantileResult(0.90, -0.10, 0.01, 0.001, 500),
        ]
        assert diagnose_distributional_pattern(results) == "approximately_uniform"

    def test_insufficient_quantiles(self):
        results = [
            QuantileResult(0.50, -0.10, 0.01, 0.001, 500),
        ]
        assert diagnose_distributional_pattern(results) == "insufficient_quantiles"
