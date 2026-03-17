from pathlib import Path

import pandas as pd

from gender_gap.standardize import psid_standardize
from gender_gap.standardize.psid_standardize import (
    _education_level,
    _education_years,
    _resolve_psid_processed_dir,
    build_psid_panel_analysis_file,
    load_psid_panel,
    standardize_psid_wave_for_gap,
    standardize_psid_2023_for_gap,
)


def _make_individuals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ER30001": 1001,
                "ER30002": 1,
                "ER32000": 1,
                "ER32022": 2,
                "ER32026": 2021,
                "ER32043": 2018,
                "ER32049": 1,
                "ER35101": 5001,
                "ER35102": 1,
                "ER35103": 10,
                "ER35104": 40,
                "ER35106": 1983,
                "ER35111": 1,
                "ER35116": 1,
                "ER35119": 1,
                "ER35127": 0,
                "ER35130": 1,
                "ER35133": 4,
                "ER35134": 1,
                "ER35135": 2,
                "ER35265": 1500,
            },
            {
                "ER30001": 1001,
                "ER30002": 2,
                "ER32000": 2,
                "ER32022": 2,
                "ER32026": 2023,
                "ER32043": 2023,
                "ER32049": 1,
                "ER35101": 5001,
                "ER35102": 2,
                "ER35103": 20,
                "ER35104": 38,
                "ER35106": 1985,
                "ER35111": 1,
                "ER35116": 1,
                "ER35119": 1,
                "ER35127": 0,
                "ER35130": 1,
                "ER35133": 2,
                "ER35134": 1,
                "ER35135": 1,
                "ER35265": 1400,
            },
        ]
    )


def _make_families() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ER82002": 5001,
                "ER82004": 36,
                "ER82018": 40,
                "ER82019": 1,
                "ER82020": 38,
                "ER82021": 2,
                "ER82022": 2,
                "ER82023": 0,
                "ER82026": 1,
                "ER82154": 1,
                "ER82156": 52,
                "ER82158": 45,
                "ER82181": 2010,
                "ER82182": 5241,
                "ER82184": 0,
                "ER82185": 35,
                "ER82186": 1,
                "ER82198": 1,
                "ER82199": 95000,
                "ER82200": 6,
                "ER82205": 0.0,
                "ER82467": 1,
                "ER82475": 48,
                "ER82477": 40,
                "ER82500": 4100,
                "ER82501": 5415,
                "ER82503": 2,
                "ER82504": 25,
                "ER82505": 3,
                "ER82517": 3,
                "ER82518": 0,
                "ER82519": 0,
                "ER82524": 32.5,
                "ER83121": 90000,
                "ER83495": 62400,
                "ER84993": 0,
                "ER84994": 1,
                "ER85120": 7,
                "ER85121": 2,
            }
        ]
    )


def _make_2021_individuals() -> pd.DataFrame:
    df = _make_individuals().rename(
        columns={
            "ER35101": "ER34901",
            "ER35102": "ER34902",
            "ER35103": "ER34903",
            "ER35104": "ER34904",
            "ER35106": "ER34906",
            "ER35111": "ER34911",
            "ER35116": "ER34916",
            "ER35119": "ER34919",
            "ER35127": "ER34927",
            "ER35130": "ER34930",
            "ER35133": "ER34933",
            "ER35134": "ER34934",
            "ER35135": "ER34935",
            "ER35265": "ER35065",
        }
    )
    df.loc[df["ER34903"] == 20, "ER32026"] = 2021
    df.loc[df["ER34903"] == 20, "ER32043"] = 2019
    return df


def _make_2021_families() -> pd.DataFrame:
    return _make_families().rename(
        columns={
            "ER82002": "ER78002",
            "ER82004": "ER78004",
            "ER82018": "ER78017",
            "ER82019": "ER78018",
            "ER82020": "ER78019",
            "ER82021": "ER78020",
            "ER82022": "ER78021",
            "ER82023": "ER78022",
            "ER82026": "ER78025",
            "ER82156": "ER78173",
            "ER82158": "ER78175",
            "ER82181": "ER78198",
            "ER82182": "ER78199",
            "ER82184": "ER78201",
            "ER82185": "ER78202",
            "ER82186": "ER78203",
            "ER82198": "ER78215",
            "ER82199": "ER78216",
            "ER82200": "ER78217",
            "ER82205": "ER78222",
            "ER82467": "ER78479",
            "ER82475": "ER78487",
            "ER82477": "ER78489",
            "ER82500": "ER78512",
            "ER82501": "ER78513",
            "ER82503": "ER78515",
            "ER82504": "ER78516",
            "ER82505": "ER78517",
            "ER82517": "ER78529",
            "ER82518": "ER78530",
            "ER82519": "ER78531",
            "ER82524": "ER78536",
            "ER83121": "ER79146",
            "ER83495": "ER79526",
            "ER84993": "ER81016",
            "ER84994": "ER81017",
            "ER85120": "ER81143",
            "ER85121": "ER81144",
        }
    )


def test_standardize_psid_2023_for_gap_builds_person_level_rows():
    standardized = standardize_psid_2023_for_gap(
        individuals=_make_individuals(),
        families=_make_families(),
    )

    assert len(standardized) == 2
    assert set(standardized["household_id"]) == {"5001"}
    assert set(standardized["female"]) == {0, 1}
    assert standardized["annual_earnings_real"].notna().all()
    assert standardized["hourly_wage_real"].notna().all()
    assert standardized["same_sex_couple_household"].eq(0).all()
    assert standardized["opposite_sex_couple_household"].eq(1).all()
    assert standardized["partner"].eq(1).all()

    spouse = standardized.loc[standardized["female"].eq(1)].iloc[0]
    assert spouse["recent_birth"] == 1
    assert spouse["recent_marriage"] == 1
    assert spouse["reproductive_stage"] == "recent_birth"
    assert spouse["race_ethnicity"] == "white_non_hispanic"
    reference = standardized.loc[standardized["female"].eq(0)].iloc[0]
    assert reference["race_ethnicity"] == "hispanic"


def test_standardize_psid_2021_wave_renames_and_sets_year():
    standardized = standardize_psid_wave_for_gap(
        survey_year=2021,
        individuals=_make_2021_individuals(),
        families=_make_2021_families(),
    )

    assert len(standardized) == 2
    assert standardized["survey_year"].eq(2021).all()
    assert standardized["calendar_year"].eq(2021).all()
    spouse = standardized.loc[standardized["female"].eq(1)].iloc[0]
    assert spouse["recent_birth"] == 1
    assert spouse["recent_marriage"] == 0
    assert spouse["race_ethnicity"] == "white_non_hispanic"


def test_standardize_psid_cleans_special_missing_codes():
    individuals = _make_individuals()
    families = _make_families()
    families.loc[0, "ER82199"] = 9_999_999
    families.loc[0, "ER83121"] = 9_999_999
    families.loc[0, "ER82205"] = 999
    families.loc[0, "ER82181"] = 9999
    families.loc[0, "ER82182"] = 9999
    families.loc[0, "ER82186"] = 9
    families.loc[0, "ER82156"] = 98
    families.loc[0, "ER82158"] = 999

    standardized = standardize_psid_2023_for_gap(individuals=individuals, families=families)
    reference = standardized.loc[standardized["female"].eq(0)].iloc[0]

    assert pd.isna(reference["wage_salary_income_real"])
    assert pd.isna(reference["annual_earnings_real"])
    assert pd.isna(reference["hourly_wage_real"])
    assert pd.isna(reference["occupation_code"])
    assert pd.isna(reference["industry_code"])
    assert pd.isna(reference["class_of_worker"])
    assert pd.isna(reference["weeks_worked"])
    assert pd.isna(reference["usual_hours_week"])


def test_standardize_psid_requires_meaningful_hours_for_fallback_hourly_wage():
    families = _make_families()
    families.loc[0, "ER82205"] = pd.NA
    families.loc[0, "ER83121"] = 150_000
    families.loc[0, "ER82156"] = 1
    families.loc[0, "ER82158"] = 1

    standardized = standardize_psid_2023_for_gap(individuals=_make_individuals(), families=families)
    reference = standardized.loc[standardized["female"].eq(0)].iloc[0]

    assert reference["annual_earnings_real"] == 150_000
    assert pd.isna(reference["hourly_wage_real"])


def test_standardize_psid_race_ethnicity_uses_hispanic_override():
    families = _make_families()
    families.loc[0, "ER85120"] = 0
    families.loc[0, "ER85121"] = 4
    families.loc[0, "ER84993"] = 1
    families.loc[0, "ER84994"] = 1

    standardized = standardize_psid_2023_for_gap(individuals=_make_individuals(), families=families)
    reference = standardized.loc[standardized["female"].eq(0)].iloc[0]
    spouse = standardized.loc[standardized["female"].eq(1)].iloc[0]

    assert reference["race_ethnicity"] == "asian"
    assert spouse["race_ethnicity"] == "hispanic"


def test_education_years_and_levels_follow_degree_rules():
    df = pd.DataFrame(
        {
            "hs_code": [3, 1, 1, 1],
            "grade_if_neither": [10, 0, 0, 0],
            "highest_year_college": [0, 0, 4, 2],
            "received_degree": [0, 0, 1, 1],
            "degree_type": [0, 0, 2, 1],
        }
    )

    years = _education_years(df)
    levels = _education_level(years)

    assert years.tolist() == [10, 12, 16, 14]
    assert levels.tolist() == ["less_than_hs", "hs_diploma", "bachelors", "some_college"]


def test_resolve_psid_processed_dir_prefers_local_then_shared(tmp_path, monkeypatch):
    local_dir = tmp_path / "paygap" / "data" / "external" / "psid"
    shared_dir = tmp_path / "shared" / "sources" / "umich" / "psid" / "main_public" / "paygap" / "processed" / "psid"
    local_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    monkeypatch.setattr(psid_standardize, "DEFAULT_LOCAL_PSID_DIR", local_dir)
    monkeypatch.setattr(psid_standardize, "DEFAULT_SHARED_PSID_PROCESSED_DIR", shared_dir)
    monkeypatch.delenv("PSID_DATA_DIR", raising=False)

    assert _resolve_psid_processed_dir() == local_dir

    local_dir.rmdir()
    assert _resolve_psid_processed_dir() == shared_dir


def test_build_and_load_psid_panel_file(tmp_path, monkeypatch):
    panel_path = tmp_path / "psid_panel.parquet"

    def _fake_standardize(survey_year, raw_dir=None, cpi_index=None, base_year=2024):
        return pd.DataFrame(
            {
                "person_id": [f"{survey_year}-1"],
                "survey_year": [survey_year],
                "calendar_year": [survey_year],
                "female": [1],
            }
        )

    monkeypatch.setattr(psid_standardize, "standardize_psid_wave_for_gap", _fake_standardize)
    build_psid_panel_analysis_file(years=(2021, 2023), output_path=panel_path)
    loaded = load_psid_panel(panel_path)

    assert panel_path.exists()
    assert loaded["survey_year"].tolist() == [2021, 2023]
