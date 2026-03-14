from pathlib import Path

import pandas as pd

from scripts.build_oaxaca_sensitivity import write_report


def test_write_report_mentions_explained_component_collapse(tmp_path: Path):
    yearly = pd.DataFrame([
        {
            "year": 2015,
            "total_gap": 0.1794,
            "explained": 0.0475,
            "unexplained": 0.1319,
            "unexplained_pct": 73.53,
            "female_coef": -0.1532,
            "pct_gap": -14.20,
        },
        {
            "year": 2021,
            "total_gap": 0.1639,
            "explained": 0.0105,
            "unexplained": 0.1535,
            "unexplained_pct": 93.62,
            "female_coef": -0.1533,
            "pct_gap": -14.21,
        },
    ])
    summary = pd.DataFrame([
        {"metric": "pre_2019_unexplained_mean", "value": 0.1284},
        {"metric": "pre_2019_total_gap_mean", "value": 0.1734},
        {"metric": "post_2019_unexplained_mean", "value": 0.1452},
        {"metric": "post_2019_total_gap_mean", "value": 0.1652},
        {"metric": "pre_2019_explained_mean", "value": 0.0450},
        {"metric": "post_2019_explained_mean", "value": 0.0200},
        {"metric": "unexplained_share_change_pp_post_minus_pre", "value": 13.85},
        {"metric": "post_share_if_pre_explained_mean_held", "value": 72.79},
        {"metric": "post_share_if_pre_unexplained_mean_held", "value": 77.73},
    ])

    output = tmp_path / "oaxaca_sensitivity.md"
    write_report(yearly, summary, output)

    text = output.read_text()
    assert "explained component" in text
    assert "Counterfactual checks" in text
    assert "Keep Oaxaca as a supplemental decomposition result" in text
