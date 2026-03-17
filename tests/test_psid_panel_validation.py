from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_psid_panel_validation.py"
_SPEC = importlib.util.spec_from_file_location("build_psid_panel_validation", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_report = _MODULE.build_report


def test_build_report_mentions_years_change_and_takeaway():
    trend = pd.DataFrame(
        [
            {
                "survey_year": 2021,
                "descriptive_hourly_gap_pct": 8.2,
                "winsorized_hourly_gap_pct": 7.9,
                "raw_annual_gap_pct": 13.0,
                "common_sample_raw_hourly_gap_pct": 21.0,
                "final_hourly_gap_pct": 15.5,
                "largest_reduction_block": "state and job sorting",
                "largest_reduction_pp": 7.2,
                "common_sample_n": 4800,
            },
            {
                "survey_year": 2023,
                "descriptive_hourly_gap_pct": 6.9,
                "winsorized_hourly_gap_pct": 6.8,
                "raw_annual_gap_pct": 11.6,
                "common_sample_raw_hourly_gap_pct": 22.5,
                "final_hourly_gap_pct": 16.6,
                "largest_reduction_block": "state and job sorting",
                "largest_reduction_pp": 8.7,
                "common_sample_n": 8565,
            },
        ]
    )

    text = build_report(trend)
    assert "# PSID Panel Validation" in text
    assert "## 2021" in text
    assert "## 2023" in text
    assert "Winsorized hourly-wage gap" in text
    assert "## Change Over Time" in text
    assert "## Takeaway" in text
