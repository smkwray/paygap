from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_psid_validation.py"
_SPEC = importlib.util.spec_from_file_location("build_psid_validation", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_report = _MODULE.build_report


def test_build_report_mentions_scope_blocks_and_takeaway():
    summary_metrics = pd.DataFrame(
        [
            {"metric": "descriptive_hourly_gap_pct", "value": 18.4},
            {"metric": "winsorized_hourly_gap_pct", "value": 17.2},
            {"metric": "winsorized_hourly_lower_bound", "value": 8.0},
            {"metric": "winsorized_hourly_upper_bound", "value": 72.0},
            {"metric": "raw_annual_gap_pct", "value": 24.7},
            {"metric": "common_sample_raw_hourly_gap_pct", "value": 21.0},
            {"metric": "final_hourly_gap_pct", "value": 12.1},
            {"metric": "largest_reduction_block", "value": "state and job sorting"},
            {"metric": "largest_reduction_pp", "value": 4.8},
            {"metric": "common_sample_n", "value": 5021},
        ]
    )
    transitions = pd.DataFrame(
        [
            {
                "from_model": "P1",
                "to_model": "P2",
                "added_block": "state and job sorting",
                "gap_before_pct": 17.0,
                "gap_after_pct": 12.2,
                "reduction_pp": 4.8,
            },
            {
                "from_model": "P4",
                "to_model": "P5",
                "added_block": "reproductive stage",
                "gap_before_pct": 13.0,
                "gap_after_pct": 12.1,
                "reduction_pp": 0.9,
            },
        ]
    )
    oaxaca_summary = pd.DataFrame(
        [
            {"component": "Total gap", "pct": 100.0},
            {"component": "Explained (endowments)", "pct": 42.0},
            {"component": "Unexplained (coefficients)", "pct": 58.0},
        ]
    )
    stage_gap = pd.DataFrame(
        [
            {"group": "recent_birth", "gap_pct": 25.0, "male_mean": 42.0, "female_mean": 31.5},
            {"group": "mother_under6", "gap_pct": 20.0, "male_mean": 40.0, "female_mean": 32.0},
            {"group": "childless_unpartnered", "gap_pct": 5.0, "male_mean": 29.0, "female_mean": 27.6},
        ]
    )

    text = build_report(summary_metrics, transitions, oaxaca_summary, stage_gap)
    assert "# PSID Validation" in text
    assert "reference persons and spouses" in text
    assert "Winsorized hourly-wage gap" in text
    assert "## Block Transitions" in text
    assert "state and job sorting" in text
    assert "## Oaxaca Snapshot" in text
    assert "## Reproductive-Stage Gaps" in text
    assert "## Takeaway" in text
