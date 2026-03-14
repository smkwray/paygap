from __future__ import annotations

import pandas as pd

from scripts.build_final_synthesis import build_report


def test_build_report_mentions_all_main_datasets():
    summary = pd.DataFrame(
        {
            "metric": [
                "acs_2023_raw_gap_pct",
                "acs_2023_m5_pct_gap",
                "acs_2023_raw_gap_ci90_low",
                "acs_2023_raw_gap_ci90_high",
                "cps_2023_raw_gap_pct",
                "cps_2023_m_full_pct_gap",
                "sipp_2023_raw_gap_pct",
                "sipp_2023_sipp3_pct_gap",
                "atus_paid_work_gap_minutes",
                "atus_housework_gap_minutes",
                "atus_childcare_gap_minutes",
                "sce_latest_expected_offer_gap",
                "sce_latest_reservation_gap",
            ],
            "value": [16.8, 13.2, 16.5, 17.1, 15.8, 17.0, 15.1, 10.9, -68.0, 31.9, 11.4, 17.5, 21.3],
            "note": [""] * 13,
        }
    )

    report = build_report(summary)
    assert "ACS 2023 raw hourly gap" in report
    assert "CPS ASEC 2023 raw hourly gap" in report
    assert "SIPP 2023 adjusted hourly gap" in report
    assert "ATUS" in report
    assert "SCE" in report
    assert "Bottom line" in report
