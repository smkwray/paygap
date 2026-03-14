from pathlib import Path

import pandas as pd

from scripts.build_m6_time_use_bridge import write_report


def test_write_report_mentions_mechanism_boundaries(tmp_path: Path):
    summary = pd.DataFrame(
        [
            {"section": "atus", "metric": "paid_work_gap_minutes", "value": -67.96, "note": ""},
            {"section": "atus", "metric": "housework_gap_minutes", "value": 31.91, "note": ""},
            {"section": "atus", "metric": "childcare_gap_minutes", "value": 11.37, "note": ""},
            {"section": "atus", "metric": "commute_gap_minutes", "value": -8.07, "note": ""},
            {"section": "atus", "metric": "unpaid_work_gap_minutes", "value": 43.28, "note": ""},
            {"section": "atus", "metric": "net_paid_plus_unpaid_gap_minutes", "value": -24.68, "note": ""},
            {"section": "acs", "metric": "m5_mean_female_coef", "value": -0.1455, "note": ""},
            {"section": "acs", "metric": "m5_latest_female_coef", "value": -0.1408, "note": ""},
            {"section": "acs", "metric": "s2_mean_ipw_worker_gap_pct", "value": 19.73, "note": ""},
            {"section": "acs", "metric": "s2_mean_expected_earnings_gap_pct", "value": 38.48, "note": ""},
            {"section": "cps", "metric": "s2_mean_ipw_worker_gap_pct", "value": 19.25, "note": ""},
            {"section": "cps", "metric": "s2_mean_expected_earnings_gap_pct", "value": 31.99, "note": ""},
        ]
    )

    output = tmp_path / "m6_time_use_bridge.md"
    write_report(summary, output)

    text = output.read_text()
    assert "not a merged ACS+ATUS regression" in text
    assert "ATUS Burden Snapshot" in text
    assert "Residual Gap Context" in text
    assert "mechanism-aware bridge" in text
