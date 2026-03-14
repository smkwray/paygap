from pathlib import Path

import pandas as pd

from scripts.build_robustness_appendix import write_report


def test_write_report_includes_core_sections(tmp_path: Path):
    summary = pd.DataFrame([
        {"section": "acs_uncertainty", "metric": "raw_gap_avg_moe90_pp", "value": 0.29, "note": ""},
        {"section": "acs_uncertainty", "metric": "m5_avg_se", "value": 0.0019, "note": ""},
        {"section": "acs_family_rebuild", "metric": "mean_m5_delta", "value": -0.0034, "note": ""},
        {"section": "acs_family_rebuild", "metric": "max_abs_m5_delta", "value": 0.0046, "note": ""},
        {"section": "acs_family_rebuild", "metric": "pooled_p5_delta", "value": -0.0042, "note": ""},
        {"section": "acs_selection", "metric": "s2_expected_gap_pct_mean", "value": 38.48, "note": ""},
        {"section": "acs_selection", "metric": "s2_ipw_worker_gap_pct_mean", "value": 19.73, "note": ""},
        {"section": "cps_selection", "metric": "s2_expected_gap_pct_mean", "value": 31.99, "note": ""},
        {"section": "cps_selection", "metric": "s2_ipw_worker_gap_pct_mean", "value": 19.25, "note": ""},
        {"section": "oaxaca", "metric": "pre_2019_unexplained_pct_mean", "value": 74.08, "note": ""},
        {"section": "oaxaca", "metric": "post_2019_unexplained_pct_mean", "value": 87.92, "note": ""},
        {"section": "context", "metric": "present_sources_count", "value": 2, "note": ""},
        {"section": "context", "metric": "manual_staging_sources_count", "value": 1, "note": ""},
    ])

    output = tmp_path / "robustness_appendix.md"
    write_report(summary, output)

    text = output.read_text()
    assert "ACS Survey Uncertainty" in text
    assert "Selection Robustness" in text
    assert "Oaxaca Stability" in text
    assert "Context Controls Status" in text
