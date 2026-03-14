from __future__ import annotations

import pandas as pd

from scripts.build_nlsy_deep_dive import build_report


def test_build_report_mentions_both_cohorts_and_takeaway():
    comparison = pd.DataFrame(
        [
            {
                "dataset": "NLSY79",
                "raw_gap_pct": 44.4,
                "skills_reduction_pp": 5.6,
                "background_reduction_pp": 0.0,
                "occupation_reduction_pp": 7.0,
                "family_reduction_pp": 0.2,
                "resources_reduction_pp": -0.9,
                "largest_reduction_block": "occupation sorting",
                "largest_reduction_pp": 7.0,
                "final_gap_pct": 34.1,
                "common_sample_n": 2890,
            },
            {
                "dataset": "NLSY97",
                "raw_gap_pct": 34.2,
                "skills_reduction_pp": 1.7,
                "background_reduction_pp": -0.1,
                "occupation_reduction_pp": 0.0,
                "family_reduction_pp": -0.2,
                "resources_reduction_pp": 6.6,
                "largest_reduction_block": "adult resources",
                "largest_reduction_pp": 6.6,
                "final_gap_pct": 31.9,
                "common_sample_n": 2486,
            },
        ]
    )
    contributions = pd.DataFrame(
        [
            {"dataset": "NLSY79", "added_block": "occupation sorting", "reduction_pp": 7.0},
            {"dataset": "NLSY79", "added_block": "skills and noncognitive traits", "reduction_pp": 5.6},
            {"dataset": "NLSY97", "added_block": "adult resources", "reduction_pp": 6.6},
            {"dataset": "NLSY97", "added_block": "skills and school achievement", "reduction_pp": 1.7},
        ]
    )

    text = build_report(comparison, contributions)
    assert "## NLSY79" in text
    assert "## NLSY97" in text
    assert "occupation sorting" in text
    assert "adult resources" in text
    assert "## Takeaway" in text
