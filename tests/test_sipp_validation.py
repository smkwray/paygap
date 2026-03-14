from __future__ import annotations

import pandas as pd

from scripts.build_sipp_validation_artifacts import build_report, build_summary


def test_build_sipp_validation_summary():
    df = pd.DataFrame(
        {
            "person_id": ["a", "b"],
            "calendar_year": [2023, 2023],
            "month": [1, 1],
            "female": [1, 0],
            "employed": [1, 0],
            "labor_force_status": ["employed", "not_in_labor_force"],
            "usual_hours_week": [40, pd.NA],
            "actual_hours_last_week": [40, pd.NA],
            "paid_hourly": [1, pd.NA],
            "hourly_wage_real": [25.0, pd.NA],
            "weekly_earnings_real": [1000.0, pd.NA],
            "overtime_indicator": [0, pd.NA],
            "multiple_jobholder": [0, 0],
            "occupation_code": ["4120", pd.NA],
            "industry_code": ["8770", pd.NA],
            "state_fips": [6, 6],
            "person_weight": [2.0, 1.0],
        }
    )

    summary = build_summary(df)
    assert set(summary.columns) == {"metric", "value"}
    assert summary.loc[summary["metric"] == "rows", "value"].iloc[0] == 2
    assert summary.loc[summary["metric"] == "hourly_wage_nonnull_share", "value"].iloc[0] == 0.5
    assert summary.loc[summary["metric"] == "weighted_mean_hourly_wage_employed", "value"].iloc[0] == 25.0


def test_build_sipp_validation_report_contains_sections():
    summary = pd.DataFrame(
        {
            "metric": ["rows", "calendar_year_min", "calendar_year_max", "female_share", "employed_share",
                       "unemployed_share", "not_in_labor_force_share", "missing_labor_force_share",
                       "weekly_earnings_nonnull_share", "hourly_wage_nonnull_share", "usual_hours_nonnull_share",
                       "actual_hours_nonnull_share", "occupation_nonnull_share_employed",
                       "industry_nonnull_share_employed", "paid_hourly_share_employed",
                       "weighted_mean_weekly_earnings_employed", "weighted_mean_hourly_wage_employed"],
            "value": [2, 2023, 2023, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1000.0, 25.0],
        }
    )

    report = build_report(summary)
    assert "# SIPP Validation" in report
    assert "## Coverage" in report
    assert "Hourly wage observed: 50.00%" in report
