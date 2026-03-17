"""Tests for ATUS reproductive-stage mechanism summaries."""

from pathlib import Path

import pandas as pd

from gender_gap.models.atus_mechanisms import (
    build_atus_mechanism_table,
    enrich_atus_with_reproductive_features,
)


def _write_inputs(tmp_path: Path):
    timeuse = pd.DataFrame(
        [
            {
                "person_id": "1",
                "calendar_year": 2024,
                "female": 1,
                "employed": 1,
                "minutes_paid_work_diary": 300,
                "minutes_work_at_home_diary": 0,
                "minutes_commute_related_travel": 20,
                "minutes_housework": 90,
                "minutes_childcare": 120,
                "minutes_eldercare": 0,
                "minutes_with_children": pd.NA,
                "person_weight": 2.0,
            },
            {
                "person_id": "2",
                "calendar_year": 2024,
                "female": 0,
                "employed": 1,
                "minutes_paid_work_diary": 420,
                "minutes_work_at_home_diary": 0,
                "minutes_commute_related_travel": 35,
                "minutes_housework": 30,
                "minutes_childcare": 40,
                "minutes_eldercare": 0,
                "minutes_with_children": pd.NA,
                "person_weight": 1.0,
            },
            {
                "person_id": "3",
                "calendar_year": 2024,
                "female": 1,
                "employed": 1,
                "minutes_paid_work_diary": 120,
                "minutes_work_at_home_diary": 0,
                "minutes_commute_related_travel": 10,
                "minutes_housework": 80,
                "minutes_childcare": 240,
                "minutes_eldercare": 0,
                "minutes_with_children": pd.NA,
                "person_weight": 1.5,
            },
        ]
    )
    respondent = pd.DataFrame(
        [
            {
                "TUCASEID": 1,
                "TUYEAR": 2024,
                "TELFS": 1,
                "TRSPPRES": 1,
                "TRTUNMPART": 0,
                "TRTSPONLY": 0,
                "TRTSPOUSE": 1,
                "TRHHCHILD": 1,
                "TRNHHCHILD": 0,
                "TROHHCHILD": 1,
                "TRCHILDNUM": 1,
                "TRYHHCHILD": 1,
                "TUFNWGTP": 2.0,
            },
            {
                "TUCASEID": 2,
                "TUYEAR": 2024,
                "TELFS": 1,
                "TRSPPRES": 0,
                "TRTUNMPART": 0,
                "TRTSPONLY": 0,
                "TRTSPOUSE": 0,
                "TRHHCHILD": 0,
                "TRNHHCHILD": 0,
                "TROHHCHILD": 0,
                "TRCHILDNUM": 0,
                "TRYHHCHILD": -1,
                "TUFNWGTP": 1.0,
            },
            {
                "TUCASEID": 3,
                "TUYEAR": 2024,
                "TELFS": 1,
                "TRSPPRES": 1,
                "TRTUNMPART": 1,
                "TRTSPONLY": 0,
                "TRTSPOUSE": 0,
                "TRHHCHILD": 1,
                "TRNHHCHILD": 0,
                "TROHHCHILD": 1,
                "TRCHILDNUM": 1,
                "TRYHHCHILD": 1,
                "TUFNWGTP": 1.5,
            },
        ]
    )
    roster = pd.DataFrame(
        [
            {"TUCASEID": 1, "TERRP": 18, "TEAGE": 30, "TESEX": 2},
            {"TUCASEID": 1, "TERRP": 20, "TEAGE": 31, "TESEX": 2},
            {"TUCASEID": 1, "TERRP": 22, "TEAGE": 2, "TESEX": 1},
            {"TUCASEID": 2, "TERRP": 18, "TEAGE": 32, "TESEX": 1},
            {"TUCASEID": 3, "TERRP": 18, "TEAGE": 28, "TESEX": 2},
            {"TUCASEID": 3, "TERRP": 21, "TEAGE": 29, "TESEX": 1},
            {"TUCASEID": 3, "TERRP": 22, "TEAGE": 0, "TESEX": 2},
        ]
    )

    processed_path = tmp_path / "atus_analysis_ready.parquet"
    respondent_path = tmp_path / "atus_respondent.parquet"
    roster_path = tmp_path / "atus_roster.parquet"
    timeuse.to_parquet(processed_path, index=False)
    respondent.to_parquet(respondent_path, index=False)
    roster.to_parquet(roster_path, index=False)
    return processed_path, respondent_path, roster_path


def test_enrich_atus_with_reproductive_features_derives_stage_and_couple_type(tmp_path: Path):
    processed_path, respondent_path, roster_path = _write_inputs(tmp_path)
    enriched = enrich_atus_with_reproductive_features(
        pd.read_parquet(processed_path),
        pd.read_parquet(respondent_path),
        pd.read_parquet(roster_path),
    ).set_index("person_id")

    assert enriched.loc["1", "same_sex_couple_household"] == 1
    assert enriched.loc["1", "reproductive_stage"] == "mother_under6"
    assert enriched.loc["2", "reproductive_stage"] == "childless_unpartnered"
    assert enriched.loc["3", "reproductive_stage"] == "recent_birth"


def test_build_atus_mechanism_table_outputs_weighted_gaps(tmp_path: Path):
    processed_path, respondent_path, roster_path = _write_inputs(tmp_path)

    result = build_atus_mechanism_table(processed_path, respondent_path, roster_path)

    overall_paid = result.loc[
        (result["reproductive_stage"] == "overall")
        & (result["metric"] == "minutes_paid_work_diary")
    ].iloc[0]

    assert overall_paid["status"] == "ok"
    assert overall_paid["n_female"] == 2
    assert overall_paid["n_male"] == 1
    assert overall_paid["female_mean_minutes"] < overall_paid["male_mean_minutes"]
    assert set(result["reproductive_stage"]) >= {"overall", "mother_under6", "recent_birth"}
