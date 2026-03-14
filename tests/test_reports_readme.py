from __future__ import annotations

import pandas as pd

from scripts.build_reports_readme import build_report


def test_build_report_mentions_requested_sections():
    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": "ACS",
                "year": 2023,
                "headline_gap_pct": 16.8,
                "adjusted_gap_pct": 13.2,
                "main_explaining_block": "job sorting",
                "block_reduction_pp": 9.3,
                "top_oaxaca_contributors": "usual_hours_week (5.0%)",
                "missing_key_factors": "firm effects",
            },
            {
                "dataset": "CPS ASEC",
                "year": 2023,
                "headline_gap_pct": 15.8,
                "adjusted_gap_pct": 17.0,
                "main_explaining_block": "job sorting",
                "block_reduction_pp": 4.1,
                "top_oaxaca_contributors": "usual_hours_week (4.0%)",
                "missing_key_factors": "commute",
            },
            {
                "dataset": "SIPP",
                "year": 2023,
                "headline_gap_pct": 15.1,
                "adjusted_gap_pct": 10.9,
                "main_explaining_block": "job sorting",
                "block_reduction_pp": 3.5,
                "top_oaxaca_contributors": "usual_hours_week (3.0%)",
                "missing_key_factors": "family controls",
            },
            {
                "dataset": "ATUS",
                "year": 2023,
                "headline_gap_pct": None,
                "adjusted_gap_pct": None,
                "main_explaining_block": "time allocation",
                "block_reduction_pp": None,
                "top_oaxaca_contributors": "paid work",
                "missing_key_factors": "wage equation",
            },
            {
                "dataset": "SCE",
                "year": 2025,
                "headline_gap_pct": None,
                "adjusted_gap_pct": None,
                "main_explaining_block": "expectations",
                "block_reduction_pp": None,
                "top_oaxaca_contributors": "reservation wage",
                "missing_key_factors": "mergeability",
            },
        ]
    )
    nlsy_summary = pd.DataFrame(
        [
            {"dataset": "NLSY79", "metric": "largest_reduction_block", "value": "occupation sorting"},
            {"dataset": "NLSY79", "metric": "largest_reduction_pp", "value": 7.0},
            {"dataset": "NLSY79", "metric": "skills_reduction_pp", "value": 5.0},
            {"dataset": "NLSY79", "metric": "background_reduction_pp", "value": 0.1},
            {"dataset": "NLSY79", "metric": "occupation_reduction_pp", "value": 8.0},
            {"dataset": "NLSY79", "metric": "final_gap_pct", "value": 33.0},
            {"dataset": "NLSY97", "metric": "largest_reduction_block", "value": "adult resources"},
            {"dataset": "NLSY97", "metric": "largest_reduction_pp", "value": 6.6},
            {"dataset": "NLSY97", "metric": "skills_reduction_pp", "value": 2.0},
            {"dataset": "NLSY97", "metric": "background_reduction_pp", "value": 0.0},
            {"dataset": "NLSY97", "metric": "occupation_reduction_pp", "value": 0.2},
            {"dataset": "NLSY97", "metric": "final_gap_pct", "value": 31.9},
        ]
    )
    future_vars = pd.DataFrame(
        [
            {"priority": 1, "variable_family": "firm effects", "why": "important"},
            {"priority": 2, "variable_family": "bargaining", "why": "important"},
        ]
    )

    text = build_report(dataset_summary, nlsy_summary, future_vars)
    assert "## ACS" in text
    assert "## CPS ASEC" in text
    assert "## SIPP" in text
    assert "## NLSY Sub-Analysis" in text
    assert "## Future Variables To Test" in text
