from __future__ import annotations

import pandas as pd

from scripts.build_sipp_snapshot import build_monthly_trend, build_snapshot_summary, write_report


def _sample_sipp_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": ["m1", "f1", "m2", "f2"],
            "calendar_year": [2023, 2023, 2023, 2023],
            "month": [1, 1, 2, 2],
            "female": [0, 1, 0, 1],
            "employed": [1, 1, 1, 0],
            "labor_force_status": ["employed", "employed", "employed", "not_in_labor_force"],
            "usual_hours_week": [40, 40, 40, pd.NA],
            "actual_hours_last_week": [40, 40, 40, pd.NA],
            "paid_hourly": [1, 1, 1, pd.NA],
            "hourly_wage_real": [30.0, 20.0, 28.0, pd.NA],
            "weekly_earnings_real": [1200.0, 800.0, 1120.0, pd.NA],
            "overtime_indicator": [0, 0, 0, pd.NA],
            "multiple_jobholder": [0, 0, 0, 0],
            "occupation_code": ["1110", "1110", "1110", pd.NA],
            "industry_code": ["2210", "2210", "2210", pd.NA],
            "state_fips": [6, 6, 6, 6],
            "person_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_build_snapshot_summary_contains_gap_metrics():
    summary = build_snapshot_summary(_sample_sipp_df()).set_index("metric")["value"]
    assert summary["rows"] == 4
    assert summary["months_covered"] == 2
    assert summary["hourly_raw_gap_pct"] > 0
    assert summary["weekly_raw_gap_pct"] > 0


def test_build_monthly_trend_returns_two_months():
    trend = build_monthly_trend(_sample_sipp_df())
    assert trend["month"].tolist() == [1, 2]
    assert set(trend.columns) == {
        "calendar_year",
        "month",
        "employed_share",
        "hourly_worker_n",
        "hourly_gap_pct",
        "weekly_worker_n",
        "weekly_gap_pct",
    }


def test_write_report_mentions_hourly_and_weekly(tmp_path):
    summary = build_snapshot_summary(_sample_sipp_df())
    trend = build_monthly_trend(_sample_sipp_df())
    path = tmp_path / "sipp_snapshot.md"
    write_report(summary, trend, path)
    text = path.read_text()
    assert "Hourly wage raw gap" in text
    assert "Weekly earnings raw gap" in text
    assert "descriptive monthly worker-gap evidence" in text
