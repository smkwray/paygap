"""Tests for Oaxaca-Blinder, Elastic Net, and model utilities."""

import numpy as np
import pandas as pd
import pytest

from gender_gap.models.dml import run_dml
from gender_gap.models.elastic_net import run_elastic_net
from gender_gap.models.oaxaca import oaxaca_blinder, oaxaca_summary_table


@pytest.fixture()
def synth_df():
    """Synthetic dataset with known gap for model testing."""
    np.random.seed(42)
    n = 400
    female = np.array([0] * 200 + [1] * 200)
    age = np.random.randint(25, 55, n).astype(float)
    hours = np.random.choice([35.0, 40.0, 45.0], n)
    children = np.random.choice([0, 1, 2], n)
    cu5 = np.where(children > 0, np.random.choice([0, 1], n), 0)
    commute = np.random.uniform(5, 60, n)
    wfh = np.random.choice([0, 1], n, p=[0.8, 0.2])
    # Wage with known gap: males ~30, females ~24
    wage = (
        20
        + 0.2 * age
        + 0.1 * hours
        - 6 * female
        - 1.0 * children * female
        + np.random.normal(0, 3, n)
    )
    wage = np.clip(wage, 5, 100)

    return pd.DataFrame({
        "female": female,
        "age": age,
        "age_sq": age ** 2,
        "hourly_wage_real": wage,
        "log_hourly_wage_real": np.log(wage),
        "usual_hours_week": hours,
        "work_from_home": wfh,
        "commute_minutes_one_way": commute,
        "number_children": children,
        "children_under_5": cu5,
        "person_weight": np.ones(n) * 100,
        "race_ethnicity": "white_non_hispanic",
        "education_level": "bachelors",
        "marital_status": "married",
        "state_fips": 6,
        "occupation_code": 1010,
        "industry_code": 7860,
        "class_of_worker": 1,
    })


class TestOaxaca:
    def test_decomposition_sums(self, synth_df):
        result = oaxaca_blinder(synth_df, controls=["age", "usual_hours_week"])
        # Explained + unexplained should approximately equal total gap
        assert abs(result.explained + result.unexplained - result.total_gap) < 0.01

    def test_gap_direction(self, synth_df):
        result = oaxaca_blinder(synth_df, controls=["age", "usual_hours_week"])
        assert result.total_gap > 0  # males earn more

    def test_contributions_dataframe(self, synth_df):
        result = oaxaca_blinder(synth_df, controls=["age", "usual_hours_week"])
        assert "variable" in result.contributions.columns
        assert "contribution" in result.contributions.columns
        assert len(result.contributions) > 0

    def test_summary_table(self, synth_df):
        result = oaxaca_blinder(synth_df, controls=["age"])
        summary = oaxaca_summary_table(result)
        assert len(summary) == 3
        assert "component" in summary.columns

    def test_percentages_sum_to_100(self, synth_df):
        result = oaxaca_blinder(synth_df, controls=["age", "usual_hours_week"])
        assert abs(result.explained_pct + result.unexplained_pct - 100.0) < 0.5


class TestElasticNet:
    def test_runs_and_returns_result(self, synth_df):
        result = run_elastic_net(
            synth_df,
            interaction_vars=["age", "usual_hours_week", "number_children"],
            cv=3,
        )
        assert result.n_obs > 0
        assert result.r_squared > 0
        assert result.female_coef < 0  # negative = women earn less

    def test_interactions_created(self, synth_df):
        result = run_elastic_net(
            synth_df,
            interaction_vars=["age", "number_children"],
            cv=3,
        )
        assert "interaction" in result.top_interactions.columns
        assert len(result.top_interactions) > 0


class TestDML:
    def test_runs_with_categorical_controls(self, synth_df):
        df = synth_df.copy()
        df["occupation_code"] = df["occupation_code"].astype(str)
        df["industry_code"] = df["industry_code"].astype(str)
        df["state_fips"] = df["state_fips"].astype(str)

        result = run_dml(
            df,
            controls=[
                "age",
                "age_sq",
                "occupation_code",
                "industry_code",
                "state_fips",
                "usual_hours_week",
            ],
            nuisance_learner="elasticnet",
            n_folds=3,
        )

        assert result.n_obs == len(df)
        assert result.std_error > 0
