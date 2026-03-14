from __future__ import annotations

import pandas as pd

from scripts.build_method_comparison import build_report


def test_build_report_mentions_estimand_caution():
    summary = pd.DataFrame(
        {
            "dataset": ["acs", "cps", "sipp"],
            "dataset_label": ["ACS", "CPS ASEC", "SIPP"],
            "year": [2023, 2023, 2023],
            "raw_gap_pct": [16.8, 15.8, 15.1],
            "ols_model": ["M5", "M_full", "SIPP3"],
            "ols_pct_gap": [13.2, 17.0, 10.9],
            "dml_pct_gap": [12.9, 16.1, 11.2],
            "oaxaca_unexplained_pct": [87.8, 75.0, 70.0],
        }
    )
    report = build_report(summary)
    assert "not identical estimands" in report
    assert "OLS adjusted gap" in report
    assert "DML adjusted gap" in report
    assert "Oaxaca unexplained" in report
    assert "elastic-net nuisance learner" in report
