"""Tests for heterogeneous gap estimation models."""

import numpy as np
import pandas as pd
import pytest

from gender_gap.models.heterogeneity import (
    HeterogeneityResult,
    estimate_heterogeneous_gaps,
    interaction_model,
    run_full_heterogeneity,
)


@pytest.fixture
def het_df():
    """DataFrame with subgroup variation for heterogeneity tests."""
    rng = np.random.RandomState(42)
    n = 500
    female = rng.binomial(1, 0.5, n)
    education = rng.choice(["hs", "ba", "grad"], n)
    age = rng.uniform(25, 55, n)
    age_sq = age ** 2

    # Create wages with different gaps by education
    base_wage = 3.0 + 0.02 * age - 0.0002 * age_sq
    ed_premium = np.where(education == "ba", 0.3, np.where(education == "grad", 0.6, 0.0))
    # Larger gap for grad, smaller for hs
    gap = np.where(education == "grad", -0.20, np.where(education == "ba", -0.12, -0.05))
    noise = rng.normal(0, 0.15, n)
    log_wage = base_wage + ed_premium + gap * female + noise

    return pd.DataFrame({
        "female": female,
        "education_level": education,
        "age": age,
        "age_sq": age_sq,
        "log_hourly_wage_real": log_wage,
        "person_weight": np.ones(n),
    })


class TestEstimateHeterogeneousGaps:
    def test_basic_output(self, het_df):
        result = estimate_heterogeneous_gaps(
            het_df, group_col="education_level"
        )
        assert isinstance(result, HeterogeneityResult)
        assert result.dimension == "education_level"
        assert len(result.subgroup_gaps) == 3  # hs, ba, grad

    def test_subgroup_columns(self, het_df):
        result = estimate_heterogeneous_gaps(
            het_df, group_col="education_level"
        )
        expected_cols = {"group", "gap", "se", "ci_lower", "ci_upper", "n"}
        assert expected_cols == set(result.subgroup_gaps.columns)

    def test_gap_signs(self, het_df):
        result = estimate_heterogeneous_gaps(
            het_df, group_col="education_level"
        )
        gaps = result.subgroup_gaps
        # All gaps should be negative (women earn less in simulation)
        for _, row in gaps.iterrows():
            assert row["gap"] < 0, f"Gap for {row['group']} should be negative"

    def test_ci_contains_point_estimate(self, het_df):
        result = estimate_heterogeneous_gaps(
            het_df, group_col="education_level"
        )
        for _, row in result.subgroup_gaps.iterrows():
            assert row["ci_lower"] <= row["gap"] <= row["ci_upper"]

    def test_small_subgroup_skipped(self, het_df):
        # Add a tiny subgroup
        het_df = het_df.copy()
        het_df.loc[het_df.index[:5], "education_level"] = "phd"
        result = estimate_heterogeneous_gaps(
            het_df, group_col="education_level"
        )
        groups = set(result.subgroup_gaps["group"])
        assert "phd" not in groups  # too few obs

    def test_log_wage_auto_computed(self, het_df):
        df = het_df.drop(columns=["log_hourly_wage_real"]).copy()
        df["hourly_wage_real"] = np.exp(het_df["log_hourly_wage_real"])
        result = estimate_heterogeneous_gaps(df, group_col="education_level")
        assert len(result.subgroup_gaps) > 0


class TestRunFullHeterogeneity:
    def test_multiple_dimensions(self, het_df):
        # Add a second dimension
        het_df = het_df.copy()
        het_df["marital_status"] = np.random.RandomState(7).choice(
            ["married", "single", "divorced"], len(het_df)
        )
        results = run_full_heterogeneity(
            het_df, dimensions=["education_level", "marital_status"]
        )
        assert "education_level" in results
        assert "marital_status" in results

    def test_missing_dimension_skipped(self, het_df):
        results = run_full_heterogeneity(
            het_df, dimensions=["education_level", "nonexistent_col"]
        )
        assert "education_level" in results
        assert "nonexistent_col" not in results

    def test_default_dimensions(self, het_df):
        results = run_full_heterogeneity(het_df)
        # Only education_level is in the defaults and present in data
        assert "education_level" in results


class TestInteractionModel:
    def test_basic_output(self, het_df):
        result = interaction_model(het_df, interact_col="education_level")
        assert isinstance(result, pd.DataFrame)
        assert "interaction" in result.columns
        assert "coef" in result.columns
        assert "se" in result.columns
        assert "pvalue" in result.columns

    def test_has_base_and_interactions(self, het_df):
        result = interaction_model(het_df, interact_col="education_level")
        interactions = result["interaction"].tolist()
        assert "female (base)" in interactions
        # Should have n_groups - 1 interaction terms
        n_interaction = sum(1 for i in interactions if i.startswith("female_x_"))
        assert n_interaction == 2  # 3 groups, drop_first=True → 2 interactions

    def test_interaction_pvalues_valid(self, het_df):
        result = interaction_model(het_df, interact_col="education_level")
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()
