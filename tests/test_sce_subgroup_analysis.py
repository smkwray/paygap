from __future__ import annotations

import pandas as pd

from scripts.build_sce_subgroup_analysis import summarize_subgroups, write_report


def test_summarize_subgroups_computes_spreads():
    tidy = pd.DataFrame(
        [
            {"date": pd.Timestamp("2025-03-01"), "measure": "expected_offer_wage", "subgroup": "Women", "value": 50.0},
            {"date": pd.Timestamp("2025-03-01"), "measure": "expected_offer_wage", "subgroup": "Men", "value": 70.0},
            {"date": pd.Timestamp("2025-03-01"), "measure": "expected_offer_wage", "subgroup": "Less than college degree", "value": 40.0},
            {"date": pd.Timestamp("2025-03-01"), "measure": "expected_offer_wage", "subgroup": "College degree or higher", "value": 90.0},
            {"date": pd.Timestamp("2025-03-01"), "measure": "expected_offer_wage", "subgroup": "<=$60K", "value": 35.0},
            {"date": pd.Timestamp("2025-03-01"), "measure": "expected_offer_wage", "subgroup": ">$60K", "value": 85.0},
            {"date": pd.Timestamp("2025-03-01"), "measure": "expected_offer_wage", "subgroup": "<=45 years", "value": 60.0},
            {"date": pd.Timestamp("2025-03-01"), "measure": "expected_offer_wage", "subgroup": ">45 years", "value": 55.0},
            {"date": pd.Timestamp("2025-07-01"), "measure": "expected_offer_wage", "subgroup": "Women", "value": 55.0},
            {"date": pd.Timestamp("2025-07-01"), "measure": "expected_offer_wage", "subgroup": "Men", "value": 72.0},
            {"date": pd.Timestamp("2025-07-01"), "measure": "expected_offer_wage", "subgroup": "Less than college degree", "value": 45.0},
            {"date": pd.Timestamp("2025-07-01"), "measure": "expected_offer_wage", "subgroup": "College degree or higher", "value": 88.0},
            {"date": pd.Timestamp("2025-07-01"), "measure": "expected_offer_wage", "subgroup": "<=$60K", "value": 40.0},
            {"date": pd.Timestamp("2025-07-01"), "measure": "expected_offer_wage", "subgroup": ">$60K", "value": 82.0},
            {"date": pd.Timestamp("2025-07-01"), "measure": "expected_offer_wage", "subgroup": "<=45 years", "value": 62.0},
            {"date": pd.Timestamp("2025-07-01"), "measure": "expected_offer_wage", "subgroup": ">45 years", "value": 57.0},
        ]
    )
    latest_df, spread_df = summarize_subgroups(tidy)
    gender = spread_df.loc[
        (spread_df["measure"] == "expected_offer_wage")
        & (spread_df["spread"] == "gender_gap_men_minus_women"),
        "latest_value",
    ].iloc[0]
    assert gender == 17.0
    assert "Women" in latest_df["subgroup"].values


def test_write_report_mentions_persistence(tmp_path):
    latest_df = pd.DataFrame(
        [
            {"measure": "expected_offer_wage", "subgroup": "Women", "latest_value": 56.0, "trailing3_mean": 57.0},
            {"measure": "expected_offer_wage", "subgroup": "Men", "latest_value": 73.0, "trailing3_mean": 74.0},
            {"measure": "expected_offer_wage", "subgroup": "<=45 years", "latest_value": 71.0, "trailing3_mean": 70.0},
            {"measure": "expected_offer_wage", "subgroup": ">45 years", "latest_value": 59.0, "trailing3_mean": 62.0},
            {"measure": "expected_offer_wage", "subgroup": "Less than college degree", "latest_value": 48.0, "trailing3_mean": 49.0},
            {"measure": "expected_offer_wage", "subgroup": "College degree or higher", "latest_value": 88.0, "trailing3_mean": 86.0},
            {"measure": "expected_offer_wage", "subgroup": "<=$60K", "latest_value": 41.0, "trailing3_mean": 40.0},
            {"measure": "expected_offer_wage", "subgroup": ">$60K", "latest_value": 83.0, "trailing3_mean": 82.0},
            {"measure": "reservation_wage", "subgroup": "Women", "latest_value": 70.0, "trailing3_mean": 70.0},
            {"measure": "reservation_wage", "subgroup": "Men", "latest_value": 91.0, "trailing3_mean": 88.0},
            {"measure": "reservation_wage", "subgroup": "<=45 years", "latest_value": 82.0, "trailing3_mean": 81.0},
            {"measure": "reservation_wage", "subgroup": ">45 years", "latest_value": 79.0, "trailing3_mean": 77.0},
            {"measure": "reservation_wage", "subgroup": "Less than college degree", "latest_value": 64.0, "trailing3_mean": 63.0},
            {"measure": "reservation_wage", "subgroup": "College degree or higher", "latest_value": 101.0, "trailing3_mean": 100.0},
            {"measure": "reservation_wage", "subgroup": "<=$60K", "latest_value": 48.0, "trailing3_mean": 50.0},
            {"measure": "reservation_wage", "subgroup": ">$60K", "latest_value": 101.0, "trailing3_mean": 98.0},
        ]
    )
    spread_df = pd.DataFrame(
        [
            {"measure": "expected_offer_wage", "spread": "gender_gap_men_minus_women", "latest_value": 17.0, "trailing3_mean": 16.3, "mean_full_sample": 15.0, "share_waves_positive": 1.0},
            {"measure": "expected_offer_wage", "spread": "education_gap_college_minus_less_than_college", "latest_value": 40.0, "trailing3_mean": 37.0, "mean_full_sample": 35.0, "share_waves_positive": 1.0},
            {"measure": "expected_offer_wage", "spread": "income_gap_high_minus_low", "latest_value": 42.0, "trailing3_mean": 41.0, "mean_full_sample": 39.0, "share_waves_positive": 1.0},
            {"measure": "expected_offer_wage", "spread": "age_gap_younger_minus_older", "latest_value": 12.0, "trailing3_mean": 8.0, "mean_full_sample": 7.0, "share_waves_positive": 1.0},
            {"measure": "reservation_wage", "spread": "gender_gap_men_minus_women", "latest_value": 21.0, "trailing3_mean": 18.0, "mean_full_sample": 17.0, "share_waves_positive": 1.0},
            {"measure": "reservation_wage", "spread": "education_gap_college_minus_less_than_college", "latest_value": 37.0, "trailing3_mean": 37.0, "mean_full_sample": 34.0, "share_waves_positive": 1.0},
            {"measure": "reservation_wage", "spread": "income_gap_high_minus_low", "latest_value": 53.0, "trailing3_mean": 48.0, "mean_full_sample": 45.0, "share_waves_positive": 1.0},
            {"measure": "reservation_wage", "spread": "age_gap_younger_minus_older", "latest_value": 3.0, "trailing3_mean": 3.0, "mean_full_sample": 4.0, "share_waves_positive": 1.0},
        ]
    )
    output = tmp_path / "sce_subgroup_analysis.md"
    write_report(latest_df, spread_df, output)
    text = output.read_text()
    assert "SCE Subgroup Analysis" in text
    assert "gender gap" in text
    assert "education and income spreads are larger than the gender spread" in text
