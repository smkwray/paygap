"""Tests for descriptive gap tables and OLS models."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from gender_gap.models.descriptive import (
    gap_by_subgroup,
    raw_gap,
    raw_gap_with_sdr,
    weighted_median_gap,
)
from gender_gap.models.ols import (
    OLSResult,
    _build_design_matrix,
    _fit_weighted_least_squares,
    female_coefficient_with_sdr,
    results_to_dataframe,
    run_sequential_ols,
)


@pytest.fixture()
def gap_df():
    """Synthetic data with known gap."""
    np.random.seed(42)
    n = 200
    female = np.array([0] * 100 + [1] * 100)
    wage = np.where(female == 0, 30.0, 24.0) + np.random.normal(0, 2, n)
    edu = np.random.choice(["bachelors", "hs_diploma", "masters"], n)
    return pd.DataFrame({
        "female": female,
        "hourly_wage_real": wage,
        "person_weight": np.ones(n) * 100,
        "education_level": edu,
        "age": np.random.randint(25, 55, n),
        "age_sq": np.random.randint(25, 55, n) ** 2,
        "race_ethnicity": "white_non_hispanic",
        "state_fips": 6,
        "occupation_code": 1010,
        "industry_code": 7860,
        "class_of_worker": 1,
        "usual_hours_week": 40,
        "work_from_home": 0,
        "commute_minutes_one_way": 25.0,
        "marital_status": "married",
        "number_children": 1,
        "children_under_5": 0,
    })


class TestRawGap:
    def test_gap_direction(self, gap_df):
        result = raw_gap(gap_df)
        assert result["male_mean"] > result["female_mean"]
        assert result["gap_dollars"] > 0
        assert result["gap_pct"] > 0

    def test_gap_magnitude(self, gap_df):
        result = raw_gap(gap_df)
        # Known gap is ~$6 (30 - 24)
        assert abs(result["gap_dollars"] - 6.0) < 2.0

    def test_counts(self, gap_df):
        result = raw_gap(gap_df)
        assert result["n_male"] == 100
        assert result["n_female"] == 100

    def test_raw_gap_with_sdr(self, gap_df):
        df = gap_df.copy()
        df["PWGTP1"] = df["person_weight"] * 0.9
        df["PWGTP2"] = np.where(df["female"] == 0, df["person_weight"] * 1.2, df["person_weight"] * 0.8)

        result = raw_gap_with_sdr(df)
        assert result["n_replicates"] == 2
        assert result["gap_pct_se"] > 0
        assert result["gap_pct_ci90_high"] > result["gap_pct_ci90_low"]


class TestGapBySubgroup:
    def test_returns_rows_per_group(self, gap_df):
        result = gap_by_subgroup(gap_df, "education_level")
        assert len(result) > 0
        assert "group" in result.columns
        assert "gap_pct" in result.columns


class TestMedianGap:
    def test_median_gap_positive(self, gap_df):
        result = weighted_median_gap(gap_df)
        assert result["gap_dollars"] > 0


class TestOLS:
    def test_sequential_ols_runs(self, gap_df):
        simple_blocks = {
            "M0": ["female"],
            "M1": ["female", "age"],
        }
        results = run_sequential_ols(
            gap_df, blocks=simple_blocks,
        )
        assert len(results) == 2
        # Female coef should be negative (women earn less)
        assert results[0].female_coef < 0

    def test_results_to_dataframe(self):
        results = [
            OLSResult("M0", -0.20, 0.02, 0.001, 0.15, 200, ["female"]),
            OLSResult("M1", -0.15, 0.02, 0.001, 0.30, 200, ["female", "age"]),
        ]
        df = results_to_dataframe(results)
        assert len(df) == 2
        assert "female_coef" in df.columns

    def test_build_design_matrix(self, gap_df):
        gap_df["log_hourly_wage_real"] = np.log(gap_df["hourly_wage_real"])
        y, X = _build_design_matrix(
            gap_df, "log_hourly_wage_real ~ female + age"
        )
        assert "female" in X.columns
        assert "age" in X.columns
        assert len(y) == len(X)

    def test_fast_wls_matches_statsmodels(self, gap_df):
        gap_df = gap_df.copy()
        gap_df["log_hourly_wage_real"] = np.log(gap_df["hourly_wage_real"])
        y, X = _build_design_matrix(
            gap_df, "log_hourly_wage_real ~ female + age"
        )
        weights = gap_df.loc[X.index, "person_weight"]

        fast = _fit_weighted_least_squares(y, X, weights)
        sm_fit = sm.WLS(y, X, weights=weights).fit()

        assert fast["n_obs"] == int(sm_fit.nobs)
        assert np.isclose(fast["params"]["female"], sm_fit.params["female"])
        assert np.isclose(fast["bse"]["female"], sm_fit.bse["female"])
        assert np.isclose(fast["r_squared"], sm_fit.rsquared)

    def test_fast_wls_drops_nonpositive_weights(self, gap_df):
        gap_df = gap_df.copy()
        gap_df["log_hourly_wage_real"] = np.log(gap_df["hourly_wage_real"])
        y, X = _build_design_matrix(
            gap_df, "log_hourly_wage_real ~ female + age"
        )
        weights = gap_df.loc[X.index, "person_weight"].copy()
        weights.iloc[0] = -1
        weights.iloc[1] = 0

        fast = _fit_weighted_least_squares(y, X, weights)
        assert fast["n_obs"] == len(X) - 2

    def test_female_coefficient_with_sdr(self, gap_df):
        df = gap_df.copy()
        df["log_hourly_wage_real"] = np.log(df["hourly_wage_real"])
        df["PWGTP1"] = df["person_weight"] * 0.95
        df["PWGTP2"] = np.where(df["female"] == 0, df["person_weight"] * 1.1, df["person_weight"] * 0.9)

        result = female_coefficient_with_sdr(
            df,
            model_name="M0",
            blocks={"M0": ["female"]},
        )

        assert result["n_replicates"] == 2
        assert result["female_coef"] < 0
        assert result["female_coef_sdr_se"] > 0
