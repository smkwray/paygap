"""Tests for resumable year-output helpers in run_all_analyses.py."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_all_analyses import (
    _acs_year_outputs_complete,
    _clean_cps_earnings,
    _cps_year_outputs_complete,
    _load_saved_acs_results,
    _load_saved_cps_results,
    _pooled_acs_outputs_complete,
    _pooled_job_terms,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_acs_year_outputs_complete_requires_full_set(tmp_path: Path):
    year_dir = tmp_path / "acs" / "2015"
    _write_csv(year_dir / "raw_gap.csv", [{"gap_pct": 10.0}])
    assert not _acs_year_outputs_complete(year_dir)

    required_files = [
        "ols_sequential.csv",
        "oaxaca.csv",
        "quantile_regression.csv",
        "heterogeneity_education_level.csv",
        "heterogeneity_race_ethnicity.csv",
        "heterogeneity_marital_status.csv",
        "heterogeneity_occupation_broad.csv",
        "heterogeneity_industry_broad.csv",
        "heterogeneity_work_from_home.csv",
        "heterogeneity_state_fips.csv",
    ]
    for name in required_files:
        _write_csv(year_dir / name, [{"x": 1}])

    assert _acs_year_outputs_complete(year_dir)


def test_load_saved_acs_results_reconstructs_trend_inputs(tmp_path: Path):
    year_dir = tmp_path / "acs" / "2015"
    _write_csv(year_dir / "raw_gap.csv", [{
        "male_mean": 30.0,
        "female_mean": 24.0,
        "gap_dollars": 6.0,
        "gap_pct": 20.0,
        "n_male": 100,
        "n_female": 100,
    }])
    _write_csv(year_dir / "ols_sequential.csv", [
        {
            "model": "M0",
            "female_coef": -0.20,
            "female_se": 0.02,
            "female_pvalue": 0.001,
            "r_squared": 0.15,
            "n_obs": 200,
        },
        {
            "model": "M5",
            "female_coef": -0.14,
            "female_se": 0.02,
            "female_pvalue": 0.001,
            "r_squared": 0.42,
            "n_obs": 200,
        },
    ])
    _write_csv(year_dir / "oaxaca.csv", [
        {"component": "Total gap", "value": 0.20, "pct": 100.0},
        {"component": "Explained (endowments)", "value": 0.05, "pct": 25.0},
        {"component": "Unexplained (coefficients)", "value": 0.15, "pct": 75.0},
    ])

    result = _load_saved_acs_results(year_dir)

    assert result["raw_gap"]["gap_pct"] == 20.0
    assert result["ols"][-1].model_name == "M5"
    assert np.isclose(result["ols"][-1].female_coef, -0.14)
    assert np.isclose(result["oaxaca"].unexplained_pct, 75.0)


def test_cps_year_outputs_complete_and_load_saved_results(tmp_path: Path):
    year_dir = tmp_path / "cps" / "2019"
    _write_csv(year_dir / "raw_gap.csv", [{
        "male_mean": 32.0,
        "female_mean": 26.0,
        "gap_dollars": 6.0,
        "gap_pct": 18.75,
        "n_male": 120,
        "n_female": 110,
    }])
    assert not _cps_year_outputs_complete(year_dir)

    _write_csv(year_dir / "ols_sequential.csv", [{
        "model": "M_full",
        "female_coef": -0.11,
        "female_se": 0.03,
        "female_pvalue": 0.002,
        "r_squared": 0.33,
        "n_obs": 230,
    }])

    assert _cps_year_outputs_complete(year_dir)

    result = _load_saved_cps_results(year_dir)
    assert result["raw_gap"]["n_female"] == 110
    assert result["ols"][0].model_name == "M_full"
    assert np.isclose(result["ols"][0].r_squared, 0.33)


def test_pooled_outputs_complete_and_job_terms(tmp_path: Path):
    pooled_dir = tmp_path / "acs_pooled"
    assert not _pooled_acs_outputs_complete(pooled_dir)

    for name in ["raw_gap_pooled.csv", "ols_pooled.csv", "oaxaca_pooled.csv"]:
        _write_csv(pooled_dir / name, [{"x": 1}])

    assert _pooled_acs_outputs_complete(pooled_dir)

    df_broad = pd.DataFrame({
        "occupation_broad": ["a"],
        "industry_broad": ["b"],
    })
    assert _pooled_job_terms(df_broad) == ("C(occupation_broad)", "C(industry_broad)")

    df_detail = pd.DataFrame({
        "occupation_code": [1010],
        "industry_code": [7860],
    })
    assert _pooled_job_terms(df_detail) == ("C(occupation_code)", "C(industry_code)")


def test_clean_cps_earnings_drops_extreme_malformed_values():
    cleaned = _clean_cps_earnings(pd.Series([-9999, 55000, 3.2e13]))
    assert cleaned.iloc[0] == 0
    assert cleaned.iloc[1] == 55000
    assert np.isnan(cleaned.iloc[2])
