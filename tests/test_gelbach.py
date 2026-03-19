"""Tests for the Gelbach (2016) exact decomposition."""

import numpy as np
import pandas as pd
import pytest

from gender_gap.models.gelbach import (
    GelbachResult,
    _build_design,
    _wls_fit,
    gelbach_decomposition,
    gelbach_to_dataframe,
)


def _make_toy_data(n=500, seed=42):
    """Build a toy dataset where the Gelbach identity can be verified.

    DGP:
        y = -0.20 * female + 0.10 * age + 0.30 * hours + 0.50 * occ_B + eps
        female is correlated with hours and occupation (the channels).
    """
    rng = np.random.default_rng(seed)
    female = rng.binomial(1, 0.5, n).astype(float)
    age = 25 + rng.normal(0, 5, n)
    # hours and occupation are correlated with female
    hours = 40 - 3 * female + rng.normal(0, 4, n)
    occ = np.where(rng.random(n) < (0.6 - 0.2 * female), "B", "A")
    eps = rng.normal(0, 0.5, n)

    y = -0.20 * female + 0.10 * age + 0.30 * hours + 0.50 * (occ == "B") + eps
    w = np.ones(n)

    return pd.DataFrame({
        "log_hourly_wage_real": y,
        "female": female,
        "age": age,
        "age_sq": age ** 2,
        "usual_hours_week": hours,
        "occupation": occ,
        "person_weight": w,
    })


class TestGelbachIdentity:
    """The exact identity: base_coef - full_coef == sum(delta_k)."""

    def test_identity_holds_numeric_only(self):
        """All-numeric blocks — identity should hold to ~1e-10."""
        df = _make_toy_data()
        blocks = {
            "base": ["female", "age"],
            "schedule": ["usual_hours_week"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        assert abs(result.identity_check) < 1e-10

    def test_identity_holds_with_categorical(self):
        """Mixed numeric + categorical blocks — identity should hold."""
        df = _make_toy_data()
        blocks = {
            "base": ["female", "age"],
            "schedule": ["usual_hours_week"],
            "job_sorting": ["C(occupation)"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        assert abs(result.identity_check) < 1e-10

    def test_total_explained_equals_sum_deltas(self):
        df = _make_toy_data()
        blocks = {
            "base": ["female", "age"],
            "schedule": ["usual_hours_week"],
            "job_sorting": ["C(occupation)"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        sum_deltas = sum(result.block_contributions.values())
        assert result.total_explained == pytest.approx(sum_deltas, abs=1e-10)


class TestGelbachDirectionality:
    """Sanity checks that the decomposition produces sensible signs."""

    def test_schedule_absorbs_gap(self):
        """Hours are negatively correlated with female, so adding hours
        to the model should reduce the female penalty (positive delta)."""
        df = _make_toy_data()
        blocks = {
            "base": ["female", "age"],
            "schedule": ["usual_hours_week"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        # Female has a negative true effect; controlling for hours
        # (which women have fewer of) should move the coef toward zero
        assert result.block_contributions["schedule"] != 0.0
        # base_coef should be more negative than full_coef
        assert result.base_coef < result.full_coef

    def test_base_coef_more_negative_than_full(self):
        """With correlated controls, base female coef picks up omitted channels."""
        df = _make_toy_data()
        blocks = {
            "base": ["female", "age"],
            "schedule": ["usual_hours_week"],
            "job_sorting": ["C(occupation)"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        assert result.base_coef < result.full_coef


class TestStringCategoricals:
    """Regression test: string categoricals must not crash or drop rows."""

    def test_string_categorical_runs(self):
        df = _make_toy_data(n=200)
        blocks = {
            "base": ["female", "age"],
            "job_sorting": ["C(occupation)"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        assert result.n_obs == 200

    def test_multiple_string_categoricals(self):
        df = _make_toy_data(n=300)
        df["region"] = np.where(df["age"] > 30, "west", "east")
        blocks = {
            "base": ["female", "age"],
            "job_sorting": ["C(occupation)"],
            "geography": ["C(region)"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        assert result.n_obs == 300
        assert abs(result.identity_check) < 1e-10


class TestMaxRowsGuard:
    def test_exceeds_max_rows_raises(self):
        df = _make_toy_data(n=500)
        with pytest.raises(ValueError, match="OOM"):
            gelbach_decomposition(df, blocks={
                "base": ["female", "age"],
                "schedule": ["usual_hours_week"],
            }, max_rows=100)

    def test_max_rows_zero_disables(self):
        df = _make_toy_data(n=500)
        result = gelbach_decomposition(df, blocks={
            "base": ["female", "age"],
            "schedule": ["usual_hours_week"],
        }, max_rows=0)
        assert result.n_obs == 500


class TestTooFewObservations:
    def test_raises_on_tiny_sample(self):
        df = _make_toy_data(n=10)
        with pytest.raises(ValueError, match="Too few"):
            gelbach_decomposition(df, blocks={
                "base": ["female", "age"],
                "schedule": ["usual_hours_week"],
            })


class TestMissingBase:
    def test_no_base_key_raises(self):
        df = _make_toy_data()
        with pytest.raises(ValueError, match="base"):
            gelbach_decomposition(df, blocks={
                "schedule": ["usual_hours_week"],
            })


class TestWeightedFit:
    """Verify WLS produces different results from OLS when weights vary."""

    def test_unequal_weights_matter(self):
        """Extreme weight imbalance should visibly shift the female coefficient."""
        df = _make_toy_data(n=500)
        blocks = {"base": ["female", "age"], "schedule": ["usual_hours_week"]}

        result_uniform = gelbach_decomposition(df, blocks=blocks)
        df2 = df.copy()
        # Use a 10:1 weight ratio so the shift is unambiguous
        df2["person_weight"] = np.where(df2["female"] == 1, 10.0, 1.0)
        result_weighted = gelbach_decomposition(df2, blocks=blocks)

        # The base coefficient should differ — sign or magnitude
        assert result_uniform.base_coef != pytest.approx(result_weighted.base_coef, abs=1e-4)


class TestGelbachToDataframe:
    def test_output_shape(self):
        df = _make_toy_data()
        blocks = {
            "base": ["female", "age"],
            "schedule": ["usual_hours_week"],
            "job_sorting": ["C(occupation)"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        table = gelbach_to_dataframe(result)
        # 2 blocks + 1 TOTAL row
        assert len(table) == 3
        assert table.iloc[-1]["block"] == "TOTAL"
        assert table.iloc[-1]["pct_of_explained"] == pytest.approx(100.0)

    def test_pct_sums_to_100(self):
        df = _make_toy_data()
        blocks = {
            "base": ["female", "age"],
            "schedule": ["usual_hours_week"],
            "job_sorting": ["C(occupation)"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        table = gelbach_to_dataframe(result)
        block_rows = table[table["block"] != "TOTAL"]
        assert block_rows["pct_of_explained"].sum() == pytest.approx(100.0, abs=0.01)


class TestSESmoke:
    """SEs should be positive and finite for well-behaved data."""

    def test_block_ses_positive_finite(self):
        df = _make_toy_data()
        blocks = {
            "base": ["female", "age"],
            "schedule": ["usual_hours_week"],
            "job_sorting": ["C(occupation)"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        for block_name, se in result.block_ses.items():
            assert se >= 0, f"SE for {block_name} is negative"
            assert np.isfinite(se), f"SE for {block_name} is not finite"


class TestTreatmentAutoPrepend:
    """Treatment variable should be auto-added to base if missing."""

    def test_treatment_not_in_base(self):
        df = _make_toy_data()
        blocks = {
            "base": ["age"],  # female intentionally omitted
            "schedule": ["usual_hours_week"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        # Should still work — female auto-prepended
        assert result.n_obs == 500
        assert abs(result.identity_check) < 1e-10


class TestNaNInCategorical:
    """NaN in a categorical column should drop those rows, not crash."""

    def test_nan_categorical_drops_rows(self):
        df = _make_toy_data(n=300)
        df.loc[0:9, "occupation"] = np.nan  # 10 rows with NaN occupation
        blocks = {
            "base": ["female", "age"],
            "job_sorting": ["C(occupation)"],
        }
        result = gelbach_decomposition(df, blocks=blocks)
        assert result.n_obs == 290


class TestZeroExplained:
    """When base and full female coefficients are the same, total_explained=0."""

    def test_to_dataframe_zero_explained(self):
        """pct_of_explained should be NaN when total_explained == 0."""
        result = GelbachResult(
            base_coef=-0.15,
            full_coef=-0.15,
            total_explained=0.0,
            block_contributions={"schedule": 0.0, "job_sorting": 0.0},
            block_ses={"schedule": 0.001, "job_sorting": 0.001},
            n_obs=500,
            r_squared_base=0.1,
            r_squared_full=0.2,
            identity_check=0.0,
        )
        table = gelbach_to_dataframe(result)
        for _, row in table[table["block"] != "TOTAL"].iterrows():
            assert pd.isna(row["pct_of_explained"])


class TestAllColumnsMissing:
    """_build_design with all requested columns absent returns empty frame."""

    def test_empty_design(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        X = _build_design(df, ["nonexistent_a", "C(nonexistent_b)"], add_constant=False)
        assert X.shape[1] == 0


class TestReproductiveBlocks:
    """Verify REPRODUCTIVE_GELBACH_BLOCKS is importable and structurally valid."""

    def test_has_base_key(self):
        from gender_gap.models.gelbach import REPRODUCTIVE_GELBACH_BLOCKS
        assert "base" in REPRODUCTIVE_GELBACH_BLOCKS

    def test_has_reproductive_key(self):
        from gender_gap.models.gelbach import REPRODUCTIVE_GELBACH_BLOCKS
        assert "reproductive" in REPRODUCTIVE_GELBACH_BLOCKS

    def test_has_job_context_key(self):
        from gender_gap.models.gelbach import REPRODUCTIVE_GELBACH_BLOCKS
        assert "job_context" in REPRODUCTIVE_GELBACH_BLOCKS

    def test_all_values_are_lists(self):
        from gender_gap.models.gelbach import REPRODUCTIVE_GELBACH_BLOCKS
        for key, val in REPRODUCTIVE_GELBACH_BLOCKS.items():
            assert isinstance(val, list), f"{key} is not a list"


class TestBuildDesign:
    def test_numeric_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        X = _build_design(df, ["x"])
        assert "const" in X.columns
        assert "x" in X.columns
        assert X.shape == (3, 2)

    def test_categorical_column(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
        X = _build_design(df, ["C(color)"])
        assert "const" in X.columns
        # drop_first=True → 2 dummies for 3 levels
        assert X.shape[1] == 3

    def test_no_constant(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        X = _build_design(df, ["x"], add_constant=False)
        assert "const" not in X.columns

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        X = _build_design(df, ["x", "missing_var"])
        assert "missing_var" not in X.columns
        assert "x" in X.columns


class TestWlsFit:
    def test_simple_ols_case(self):
        """With uniform weights, WLS should match OLS."""
        rng = np.random.default_rng(99)
        n = 200
        x = rng.normal(0, 1, n)
        y = 2.0 + 3.0 * x + rng.normal(0, 0.5, n)
        X = pd.DataFrame({"const": np.ones(n), "x": x})
        w = np.ones(n)
        beta, se, r2 = _wls_fit(y, X, w)
        assert beta[0] == pytest.approx(2.0, abs=0.2)
        assert beta[1] == pytest.approx(3.0, abs=0.2)
        assert r2 > 0.8
