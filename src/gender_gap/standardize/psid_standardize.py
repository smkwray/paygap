"""Standardize PSID main-panel public data for paygap validation."""

from __future__ import annotations

import logging
import os
import re
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from gender_gap.features.reproductive import add_reproductive_features
from gender_gap.settings import shared_source_path

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_PSID_DIR = Path("data/external/psid")
DEFAULT_SHARED_PSID_RAW_DIR = shared_source_path(
    "umich",
    "psid",
    "main_public",
    "paygap",
    "raw",
    "psid",
)
DEFAULT_SHARED_PSID_PROCESSED_DIR = shared_source_path(
    "umich",
    "psid",
    "main_public",
    "paygap",
    "processed",
    "psid",
)
DEFAULT_PROCESSED_FILENAME = "psid_2023_analysis_ready.parquet"
DEFAULT_PANEL_FILENAME = "psid_2021_2023_analysis_ready.parquet"

_PSID_MARITAL_STATUS = {
    1: "married",
    2: "never_married",
    3: "widowed",
    4: "divorced",
    5: "separated",
}

_PSID_REQUIRED_INDIVIDUAL_VARS = [
    "ER30001",
    "ER30002",
    "ER32000",
    "ER32022",
    "ER32026",
    "ER32043",
    "ER32049",
    "ER35101",
    "ER35102",
    "ER35103",
    "ER35104",
    "ER35106",
    "ER35111",
    "ER35116",
    "ER35119",
    "ER35127",
    "ER35130",
    "ER35133",
    "ER35134",
    "ER35135",
    "ER35265",
]

_PSID_REQUIRED_FAMILY_VARS = [
    "ER82002",
    "ER82004",
    "ER82018",
    "ER82019",
    "ER82020",
    "ER82021",
    "ER82022",
    "ER82023",
    "ER82026",
    "ER82154",
    "ER82156",
    "ER82158",
    "ER82181",
    "ER82182",
    "ER82184",
    "ER82185",
    "ER82186",
    "ER82198",
    "ER82199",
    "ER82200",
    "ER82205",
    "ER82467",
    "ER82475",
    "ER82477",
    "ER82500",
    "ER82501",
    "ER82503",
    "ER82504",
    "ER82505",
    "ER82517",
    "ER82518",
    "ER82519",
    "ER82524",
    "ER83121",
    "ER83495",
    "ER84993",
    "ER84994",
    "ER85120",
    "ER85121",
]

_PSID_WAVE_CONFIG = {
    2021: {
        "family_zip": "psid_family_2021.zip",
        "family_setup": "FAM2021ER.sas",
        "family_text": "FAM2021ER.txt",
        "individual_setup": "IND2023ER.sas",
        "individual_text": "IND2023ER.txt",
        "individual_vars": [
            "ER30001",
            "ER30002",
            "ER32000",
            "ER32022",
            "ER32026",
            "ER32043",
            "ER32049",
            "ER34901",
            "ER34902",
            "ER34903",
            "ER34904",
            "ER34906",
            "ER34911",
            "ER34916",
            "ER34919",
            "ER34927",
            "ER34930",
            "ER34933",
            "ER34934",
            "ER34935",
            "ER35065",
        ],
        "family_vars": [
            "ER78002",
            "ER78004",
            "ER78017",
            "ER78018",
            "ER78019",
            "ER78020",
            "ER78021",
            "ER78022",
            "ER78025",
            "ER78173",
            "ER78175",
            "ER78198",
            "ER78199",
            "ER78201",
            "ER78202",
            "ER78203",
            "ER78215",
            "ER78216",
            "ER78217",
            "ER78222",
            "ER78479",
            "ER78487",
            "ER78489",
            "ER78512",
            "ER78513",
            "ER78515",
            "ER78516",
            "ER78517",
            "ER78529",
            "ER78530",
            "ER78531",
            "ER78536",
            "ER79146",
            "ER79526",
            "ER81016",
            "ER81017",
            "ER81143",
            "ER81144",
        ],
        "individual_rename": {
            "ER34901": "ER35101",
            "ER34902": "ER35102",
            "ER34903": "ER35103",
            "ER34904": "ER35104",
            "ER34906": "ER35106",
            "ER34911": "ER35111",
            "ER34916": "ER35116",
            "ER34919": "ER35119",
            "ER34927": "ER35127",
            "ER34930": "ER35130",
            "ER34933": "ER35133",
            "ER34934": "ER35134",
            "ER34935": "ER35135",
            "ER35065": "ER35265",
        },
        "family_rename": {
            "ER78002": "ER82002",
            "ER78004": "ER82004",
            "ER78017": "ER82018",
            "ER78018": "ER82019",
            "ER78019": "ER82020",
            "ER78020": "ER82021",
            "ER78021": "ER82022",
            "ER78022": "ER82023",
            "ER78025": "ER82026",
            "ER78173": "ER82156",
            "ER78175": "ER82158",
            "ER78198": "ER82181",
            "ER78199": "ER82182",
            "ER78201": "ER82184",
            "ER78202": "ER82185",
            "ER78203": "ER82186",
            "ER78215": "ER82198",
            "ER78216": "ER82199",
            "ER78217": "ER82200",
            "ER78222": "ER82205",
            "ER78479": "ER82467",
            "ER78487": "ER82475",
            "ER78489": "ER82477",
            "ER78512": "ER82500",
            "ER78513": "ER82501",
            "ER78515": "ER82503",
            "ER78516": "ER82504",
            "ER78517": "ER82505",
            "ER78529": "ER82517",
            "ER78530": "ER82518",
            "ER78531": "ER82519",
            "ER78536": "ER82524",
            "ER79146": "ER83121",
            "ER79526": "ER83495",
            "ER81016": "ER84993",
            "ER81017": "ER84994",
            "ER81143": "ER85120",
            "ER81144": "ER85121",
        },
    },
    2023: {
        "family_zip": "psid_family_2023.zip",
        "family_setup": "FAM2023ER.sas",
        "family_text": "FAM2023ER.txt",
        "individual_setup": "IND2023ER.sas",
        "individual_text": "IND2023ER.txt",
        "individual_vars": _PSID_REQUIRED_INDIVIDUAL_VARS,
        "family_vars": _PSID_REQUIRED_FAMILY_VARS,
        "individual_rename": {},
        "family_rename": {},
    },
}


def _resolve_psid_processed_dir() -> Path:
    env_dir = os.environ.get("PSID_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    if DEFAULT_LOCAL_PSID_DIR.exists():
        return DEFAULT_LOCAL_PSID_DIR
    if DEFAULT_SHARED_PSID_PROCESSED_DIR.exists():
        return DEFAULT_SHARED_PSID_PROCESSED_DIR
    return DEFAULT_LOCAL_PSID_DIR


def _resolve_psid_raw_dir() -> Path:
    env_dir = os.environ.get("PSID_RAW_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return DEFAULT_SHARED_PSID_RAW_DIR


def load_psid_2023(path: Path | None = None) -> pd.DataFrame:
    """Load the processed 2023 PSID validation extract."""
    if path is None:
        path = _resolve_psid_processed_dir() / DEFAULT_PROCESSED_FILENAME
        if not path.exists():
            shared_path = DEFAULT_SHARED_PSID_PROCESSED_DIR / DEFAULT_PROCESSED_FILENAME
            if shared_path.exists():
                path = shared_path
    logger.info("Loading PSID 2023 from %s", path)
    return pd.read_parquet(path)


def load_psid_panel(path: Path | None = None) -> pd.DataFrame:
    """Load the processed 2021/2023 PSID validation panel extract."""
    if path is None:
        path = _resolve_psid_processed_dir() / DEFAULT_PANEL_FILENAME
        if not path.exists():
            shared_path = DEFAULT_SHARED_PSID_PROCESSED_DIR / DEFAULT_PANEL_FILENAME
            if shared_path.exists():
                path = shared_path
    logger.info("Loading PSID panel from %s", path)
    return pd.read_parquet(path)


def load_psid_wave_raw(
    survey_year: int,
    raw_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the narrow raw PSID slices needed for a given survey wave."""
    root = Path(raw_dir) if raw_dir is not None else _resolve_psid_raw_dir()
    config = _wave_config(survey_year)
    individuals = _read_psid_subset(
        root / "psid_cross_year_individual_1968_2023.zip",
        config["individual_setup"],
        config["individual_text"],
        config["individual_vars"],
    )
    families = _read_psid_subset(
        root / config["family_zip"],
        config["family_setup"],
        config["family_text"],
        config["family_vars"],
    )
    return _canonicalize_wave_columns(individuals, families, survey_year)


def load_psid_2023_raw(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the narrow raw PSID slices needed for the paygap adapter."""
    return load_psid_wave_raw(2023, raw_dir=raw_dir)


def build_psid_2023_analysis_file(
    output_path: Path | None = None,
    raw_dir: Path | None = None,
    cpi_index: dict[int, float] | None = None,
    base_year: int = 2024,
) -> Path:
    """Build and persist the 2023 PSID analysis-ready extract."""
    if output_path is None:
        output_path = DEFAULT_SHARED_PSID_PROCESSED_DIR / DEFAULT_PROCESSED_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized = standardize_psid_2023_for_gap(
        raw_dir=raw_dir,
        cpi_index=cpi_index,
        base_year=base_year,
    )
    standardized.to_parquet(output_path, index=False)
    logger.info("Wrote PSID 2023 analysis-ready file to %s", output_path)
    return output_path


def build_psid_panel_analysis_file(
    years: tuple[int, ...] = (2021, 2023),
    output_path: Path | None = None,
    raw_dir: Path | None = None,
    cpi_index: dict[int, float] | None = None,
    base_year: int = 2024,
) -> Path:
    """Build and persist a stacked PSID validation panel for supported waves."""
    if output_path is None:
        output_path = DEFAULT_SHARED_PSID_PROCESSED_DIR / DEFAULT_PANEL_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        standardize_psid_wave_for_gap(
            survey_year=year,
            raw_dir=raw_dir,
            cpi_index=cpi_index,
            base_year=base_year,
        )
        for year in years
    ]
    pd.concat(frames, ignore_index=True).to_parquet(output_path, index=False)
    logger.info("Wrote PSID panel analysis-ready file to %s", output_path)
    return output_path


def standardize_psid_wave_for_gap(
    survey_year: int,
    individuals: pd.DataFrame | None = None,
    families: pd.DataFrame | None = None,
    raw_dir: Path | None = None,
    cpi_index: dict[int, float] | None = None,
    base_year: int = 2024,
) -> pd.DataFrame:
    """Standardize a supported PSID main-panel wave into paygap's annual schema."""
    if individuals is None or families is None:
        individuals, families = load_psid_wave_raw(survey_year, raw_dir=raw_dir)
    else:
        individuals, families = _canonicalize_wave_columns(individuals, families, survey_year)

    people = _prepare_individual_frame(individuals, survey_year=survey_year)
    family = _prepare_family_frame(families)

    ref = _assemble_person_rows(
        family=family,
        people=people,
        relation_code=10,
        sex_col="reference_person_sex",
        age_col="reference_person_age",
        weeks_col="rp_weeks_worked",
        hours_col="rp_hours_worked",
        occ_col="rp_occupation_code",
        ind_col="rp_industry_code",
        commute_minutes_col="rp_commute_minutes",
        work_from_home_col="rp_work_from_home",
        class_col="rp_class_of_worker",
        pay_type_col="rp_pay_type",
        hourly_col="rp_hourly_rate",
        salary_col="rp_salary_amount",
        salary_per_col="rp_salary_per",
        wage_salary_col="rp_wage_salary_income",
        survey_year=survey_year,
    )
    spouse = _assemble_person_rows(
        family=family,
        people=people,
        relation_code=20,
        sex_col="spouse_sex",
        age_col="spouse_age",
        weeks_col="sp_weeks_worked",
        hours_col="sp_hours_worked",
        occ_col="sp_occupation_code",
        ind_col="sp_industry_code",
        commute_minutes_col="sp_commute_minutes",
        work_from_home_col="sp_work_from_home",
        class_col="sp_class_of_worker",
        pay_type_col="sp_pay_type",
        hourly_col="sp_hourly_rate",
        salary_col="sp_salary_amount",
        salary_per_col="sp_salary_per",
        wage_salary_col="sp_wage_salary_income",
        survey_year=survey_year,
    )
    spouse = spouse.loc[spouse["partner"].eq(1)].copy()

    out = pd.concat([ref, spouse], ignore_index=True, sort=False)
    out["annual_hours"] = out["weeks_worked"] * out["usual_hours_week"]
    out["wage_salary_income_real"] = _inflate_to_base_year(
        out["wage_salary_income_real"],
        survey_year=survey_year,
        cpi_index=cpi_index,
        base_year=base_year,
    )
    out["annual_earnings_real"] = _inflate_to_base_year(
        out["annual_earnings_real"],
        survey_year=survey_year,
        cpi_index=cpi_index,
        base_year=base_year,
    )
    out["hourly_wage_real"] = _coerce_positive(out["hourly_wage_real"])
    out["wage_salary_income_real"] = _coerce_positive(out["wage_salary_income_real"])
    out["annual_earnings_real"] = _coerce_positive(out["annual_earnings_real"])
    fallback_hours = out["annual_hours"].where(out["annual_hours"].ge(52))
    fallback_hourly = out["annual_earnings_real"] / fallback_hours
    out["hourly_wage_real"] = out["hourly_wage_real"].fillna(fallback_hourly)
    out["age_sq"] = out["age"] ** 2
    out["self_employed"] = _self_employed_flag(out["class_of_worker"])
    out = add_reproductive_features(out)
    return out


def standardize_psid_2023_for_gap(
    individuals: pd.DataFrame | None = None,
    families: pd.DataFrame | None = None,
    raw_dir: Path | None = None,
    cpi_index: dict[int, float] | None = None,
    base_year: int = 2024,
) -> pd.DataFrame:
    """Standardize the 2023 PSID main panel into paygap's annual schema."""
    return standardize_psid_wave_for_gap(
        survey_year=2023,
        individuals=individuals,
        families=families,
        raw_dir=raw_dir,
        cpi_index=cpi_index,
        base_year=base_year,
    )


def _prepare_individual_frame(individuals: pd.DataFrame, survey_year: int = 2023) -> pd.DataFrame:
    out = individuals.copy()
    out = out.loc[out["ER35101"].gt(0)].copy()
    out["household_id"] = out["ER35101"].astype("Int64").astype(str)
    out["person_id"] = (
        out["ER30001"].astype("Int64").astype(str).str.zfill(4)
        + "-"
        + out["ER30002"].astype("Int64").astype(str).str.zfill(3)
    )
    out["relation_code"] = pd.to_numeric(out["ER35103"], errors="coerce")
    out["age"] = _clean_psid_age(out["ER35104"])
    out["female"] = (pd.to_numeric(out["ER32000"], errors="coerce") == 2).astype("Int64")
    out["person_weight"] = _clean_psid_positive_measure(out["ER35265"])
    out["employment_status"] = pd.to_numeric(out["ER35116"], errors="coerce")
    out["hs_code"] = pd.to_numeric(out["ER35119"], errors="coerce")
    out["grade_if_neither"] = pd.to_numeric(out["ER35127"], errors="coerce")
    out["attended_college"] = pd.to_numeric(out["ER35130"], errors="coerce")
    out["highest_year_college"] = pd.to_numeric(out["ER35133"], errors="coerce")
    out["received_degree"] = pd.to_numeric(out["ER35134"], errors="coerce")
    out["degree_type"] = pd.to_numeric(out["ER35135"], errors="coerce")
    out["youngest_child_birth_year"] = _clean_psid_year(out["ER32026"], survey_year=survey_year)
    out["most_recent_marriage_year"] = _clean_psid_year(out["ER32043"], survey_year=survey_year)
    out["last_known_marital_status"] = pd.to_numeric(out["ER32049"], errors="coerce")
    out["education_years"] = _education_years(out)
    out["education_level"] = _education_level(out["education_years"])
    out["recent_birth"] = _derive_recent_birth(out["youngest_child_birth_year"], survey_year=survey_year)
    out["recent_marriage"] = _derive_recent_marriage(out["most_recent_marriage_year"], survey_year=survey_year)
    return out


def _prepare_family_frame(families: pd.DataFrame) -> pd.DataFrame:
    out = families.copy()
    out["household_id"] = pd.to_numeric(out["ER82002"], errors="coerce").astype("Int64").astype(str)
    out["state_fips"] = _clean_psid_code(out["ER82004"], invalid_values={0})
    out["reference_person_age"] = _clean_psid_age(out["ER82018"])
    out["reference_person_sex"] = pd.to_numeric(out["ER82019"], errors="coerce")
    out["spouse_age"] = _clean_psid_age(out["ER82020"])
    out["spouse_sex"] = pd.to_numeric(out["ER82021"], errors="coerce")
    out["number_children"] = _clean_psid_count(out["ER82022"]).fillna(0)
    out["youngest_child_age"] = _clean_psid_age(out["ER82023"])
    out["psid_marital_status_code"] = pd.to_numeric(out["ER82026"], errors="coerce")
    out["spouse_spanish_descent"] = _clean_psid_code(out["ER84993"], invalid_values={0, 9})
    out["spouse_race_mention_1"] = _clean_psid_code(out["ER84994"], invalid_values={0, 9})
    out["reference_spanish_descent"] = _clean_psid_code(out["ER85120"], invalid_values={0, 9})
    out["reference_race_mention_1"] = _clean_psid_code(out["ER85121"], invalid_values={0, 9})
    out["partner"] = (
        pd.to_numeric(out["ER82467"], errors="coerce").eq(1)
        | out["spouse_sex"].isin([1, 2])
    ).astype(int)
    out["same_sex_couple_household"] = (
        out["partner"].eq(1) & out["reference_person_sex"].eq(out["spouse_sex"])
    ).astype(int)
    out["opposite_sex_couple_household"] = (
        out["partner"].eq(1) & out["reference_person_sex"].ne(out["spouse_sex"])
    ).astype(int)
    out["couple_type"] = np.select(
        [
            out["same_sex_couple_household"].eq(1),
            out["opposite_sex_couple_household"].eq(1),
        ],
        ["same_sex", "opposite_sex"],
        default="unpartnered",
    )
    out["cplt"] = np.select(
        [
            out["same_sex_couple_household"].eq(1) & out["psid_marital_status_code"].eq(1),
            out["same_sex_couple_household"].eq(1),
            out["opposite_sex_couple_household"].eq(1) & out["psid_marital_status_code"].eq(1),
            out["opposite_sex_couple_household"].eq(1),
        ],
        [2, 4, 1, 3],
        default=0,
    )
    out["has_own_child"] = out["number_children"].gt(0).astype(int)
    out["own_child_under6"] = (
        out["number_children"].gt(0) & out["youngest_child_age"].between(0, 5, inclusive="both")
    ).astype(int)
    out["own_child_6_17_only"] = (
        out["number_children"].gt(0) & out["youngest_child_age"].between(6, 17, inclusive="both")
    ).astype(int)
    out["children_under_5"] = (
        out["number_children"].gt(0) & out["youngest_child_age"].between(0, 4, inclusive="both")
    ).astype(int)
    out["rp_race_ethnicity"] = _recode_psid_race_ethnicity(
        out["reference_spanish_descent"],
        out["reference_race_mention_1"],
    )
    out["sp_race_ethnicity"] = _recode_psid_race_ethnicity(
        out["spouse_spanish_descent"],
        out["spouse_race_mention_1"],
    )
    out["noc"] = out["number_children"]
    out["paoc"] = np.select(
        [out["own_child_under6"].eq(1), out["own_child_6_17_only"].eq(1)],
        [1, 2],
        default=0,
    )
    rename_map = {
        "ER82156": "rp_weeks_worked",
        "ER82158": "rp_hours_worked",
        "ER82181": "rp_occupation_code",
        "ER82182": "rp_industry_code",
        "ER82184": "rp_work_from_home",
        "ER82185": "rp_commute_minutes",
        "ER82186": "rp_class_of_worker",
        "ER82198": "rp_pay_type",
        "ER82199": "rp_salary_amount",
        "ER82200": "rp_salary_per",
        "ER82205": "rp_hourly_rate",
        "ER83121": "rp_wage_salary_income",
        "ER82475": "sp_weeks_worked",
        "ER82477": "sp_hours_worked",
        "ER82500": "sp_occupation_code",
        "ER82501": "sp_industry_code",
        "ER82503": "sp_work_from_home",
        "ER82504": "sp_commute_minutes",
        "ER82505": "sp_class_of_worker",
        "ER82517": "sp_pay_type",
        "ER82518": "sp_salary_amount",
        "ER82519": "sp_salary_per",
        "ER82524": "sp_hourly_rate",
        "ER83495": "sp_wage_salary_income",
    }
    out = out.rename(columns=rename_map)
    numeric_cols = list(rename_map.values())
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    out["rp_weeks_worked"] = _clean_psid_weeks(out["rp_weeks_worked"])
    out["sp_weeks_worked"] = _clean_psid_weeks(out["sp_weeks_worked"])
    out["rp_hours_worked"] = _clean_psid_hours(out["rp_hours_worked"])
    out["sp_hours_worked"] = _clean_psid_hours(out["sp_hours_worked"])
    out["rp_occupation_code"] = _clean_psid_code(out["rp_occupation_code"], invalid_values={0})
    out["sp_occupation_code"] = _clean_psid_code(out["sp_occupation_code"], invalid_values={0})
    out["rp_industry_code"] = _clean_psid_code(out["rp_industry_code"], invalid_values={0})
    out["sp_industry_code"] = _clean_psid_code(out["sp_industry_code"], invalid_values={0})
    out["rp_commute_minutes"] = _clean_psid_commute(out["rp_commute_minutes"])
    out["sp_commute_minutes"] = _clean_psid_commute(out["sp_commute_minutes"])
    out["rp_class_of_worker"] = _clean_psid_code(out["rp_class_of_worker"], invalid_values={8, 9})
    out["sp_class_of_worker"] = _clean_psid_code(out["sp_class_of_worker"], invalid_values={8, 9})
    out["rp_pay_type"] = _clean_psid_code(out["rp_pay_type"], invalid_values={0, 8, 9})
    out["sp_pay_type"] = _clean_psid_code(out["sp_pay_type"], invalid_values={0, 8, 9})
    out["rp_salary_per"] = _clean_psid_code(out["rp_salary_per"], invalid_values={0, 8, 9})
    out["sp_salary_per"] = _clean_psid_code(out["sp_salary_per"], invalid_values={0, 8, 9})
    out["rp_salary_amount"] = _clean_psid_positive_measure(out["rp_salary_amount"])
    out["sp_salary_amount"] = _clean_psid_positive_measure(out["sp_salary_amount"])
    out["rp_hourly_rate"] = _clean_psid_positive_measure(out["rp_hourly_rate"])
    out["sp_hourly_rate"] = _clean_psid_positive_measure(out["sp_hourly_rate"])
    out["rp_wage_salary_income"] = _clean_psid_positive_measure(out["rp_wage_salary_income"])
    out["sp_wage_salary_income"] = _clean_psid_positive_measure(out["sp_wage_salary_income"])
    out["marital_status"] = out["psid_marital_status_code"].map(_PSID_MARITAL_STATUS).fillna("other")
    return out


def _assemble_person_rows(
    family: pd.DataFrame,
    people: pd.DataFrame,
    relation_code: int,
    sex_col: str,
    age_col: str,
    weeks_col: str,
    hours_col: str,
    occ_col: str,
    ind_col: str,
    commute_minutes_col: str,
    work_from_home_col: str,
    class_col: str,
    pay_type_col: str,
    hourly_col: str,
    salary_col: str,
    salary_per_col: str,
    wage_salary_col: str,
    survey_year: int,
) -> pd.DataFrame:
    match = people.loc[people["relation_code"].eq(relation_code)].copy()
    match = match.sort_values(["household_id", "person_weight"], ascending=[True, False])
    match = match.drop_duplicates(subset=["household_id"])
    merged = family.merge(
        match[
            [
                "household_id",
                "person_id",
                "female",
                "age",
                "person_weight",
                "education_years",
                "education_level",
                "recent_birth",
                "recent_marriage",
            ]
        ],
        on="household_id",
        how="left",
    )
    merged["person_id"] = merged["person_id"].fillna(
        merged["household_id"].astype(str) + ("-RP" if relation_code == 10 else "-SP")
    )
    family_female = (pd.to_numeric(merged[sex_col], errors="coerce") == 2).astype("Int64")
    merged["female"] = merged["female"].fillna(family_female)
    merged["age"] = merged["age"].fillna(pd.to_numeric(merged[age_col], errors="coerce"))
    merged["data_source"] = "PSID"
    merged["survey_year"] = survey_year
    merged["calendar_year"] = survey_year
    merged["household_id"] = merged["household_id"].astype(str)
    merged["acs_serialno"] = pd.NA
    merged["acs_sporder"] = pd.NA
    merged["occupation_code"] = merged[occ_col]
    merged["industry_code"] = merged[ind_col]
    merged["class_of_worker"] = merged[class_col]
    merged["weeks_worked"] = merged[weeks_col]
    merged["usual_hours_week"] = _clean_psid_hours(merged[hours_col])
    merged["work_from_home"] = pd.to_numeric(merged[work_from_home_col], errors="coerce").between(
        1, 7, inclusive="both"
    ).astype(int)
    merged["commute_minutes_one_way"] = _clean_psid_commute(merged[commute_minutes_col])
    merged["state_fips"] = pd.to_numeric(merged["state_fips"], errors="coerce")
    merged["residence_puma"] = pd.NA
    merged["place_of_work_state"] = pd.NA
    merged["place_of_work_puma"] = pd.NA
    merged["wage_salary_income_real"] = _clean_psid_positive_measure(merged[wage_salary_col])
    merged["annual_earnings_real"] = merged["wage_salary_income_real"].copy()
    merged["hourly_wage_real"] = _clean_psid_positive_measure(merged[hourly_col])
    merged["race_ethnicity"] = merged["rp_race_ethnicity"] if relation_code == 10 else merged["sp_race_ethnicity"]
    merged["marital_status"] = merged["marital_status"].fillna("other")
    merged["fer"] = merged["recent_birth"].fillna(0).astype(int)
    merged["marhm"] = merged["recent_marriage"].fillna(0).astype(int)
    merged["partner"] = merged["partner"].fillna(0).astype(int)
    merged["relshipp"] = relation_code
    merged["annual_earnings_real"] = merged["annual_earnings_real"].fillna(
        _annualize_salary(
            salary_amount=_clean_psid_positive_measure(merged[salary_col]),
            salary_per=_clean_psid_code(merged[salary_per_col], invalid_values={0, 8, 9}),
            pay_type=_clean_psid_code(merged[pay_type_col], invalid_values={0, 8, 9}),
            weeks_worked=_clean_psid_weeks(merged[weeks_col]),
            usual_hours=_clean_psid_hours(merged[hours_col]),
            hourly_rate=_clean_psid_positive_measure(merged[hourly_col]),
        )
    )
    return merged[
        [
            "person_id",
            "household_id",
            "data_source",
            "survey_year",
            "calendar_year",
            "acs_serialno",
            "acs_sporder",
            "female",
            "age",
            "education_years",
            "education_level",
            "race_ethnicity",
            "marital_status",
            "fer",
            "marhm",
            "cplt",
            "partner",
            "relshipp",
            "paoc",
            "noc",
            "number_children",
            "children_under_5",
            "recent_birth",
            "recent_marriage",
            "has_own_child",
            "own_child_under6",
            "own_child_6_17_only",
            "same_sex_couple_household",
            "opposite_sex_couple_household",
            "couple_type",
            "occupation_code",
            "industry_code",
            "class_of_worker",
            "weeks_worked",
            "usual_hours_week",
            "work_from_home",
            "commute_minutes_one_way",
            "state_fips",
            "residence_puma",
            "place_of_work_state",
            "place_of_work_puma",
            "hourly_wage_real",
            "annual_earnings_real",
            "wage_salary_income_real",
            "person_weight",
        ]
    ].copy()


def _read_psid_subset(
    zip_path: Path,
    setup_name: str,
    data_name: str,
    variables: list[str],
) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        setup_text = zf.read(setup_name).decode("latin-1")
        data_text = zf.read(data_name).decode("latin-1")
    colspecs = _parse_sas_input_specs(setup_text, variables)
    names = list(colspecs)
    return pd.read_fwf(
        StringIO(data_text),
        colspecs=list(colspecs.values()),
        names=names,
    )


def _wave_config(survey_year: int) -> dict:
    if survey_year not in _PSID_WAVE_CONFIG:
        raise ValueError(f"Unsupported PSID survey year: {survey_year}")
    return _PSID_WAVE_CONFIG[survey_year]


def _canonicalize_wave_columns(
    individuals: pd.DataFrame,
    families: pd.DataFrame,
    survey_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _wave_config(survey_year)
    return (
        individuals.rename(columns=config["individual_rename"]).copy(),
        families.rename(columns=config["family_rename"]).copy(),
    )


def _parse_sas_input_specs(setup_text: str, variables: list[str]) -> dict[str, tuple[int, int]]:
    match = re.search(r"INPUT\s+(.*?)\s*;", setup_text, re.S)
    if not match:
        raise ValueError("Could not locate INPUT block in PSID setup file.")
    input_block = match.group(1)
    specs: dict[str, tuple[int, int]] = {}
    for name, start, end in re.findall(r"([A-Z0-9]+)\s+(\d+)\s*-\s*(\d+)", input_block):
        if name in variables:
            specs[name] = (int(start) - 1, int(end))
    missing = sorted(set(variables) - set(specs))
    if missing:
        raise ValueError(f"Missing PSID variables in setup file: {missing}")
    return specs


def _education_years(df: pd.DataFrame) -> pd.Series:
    years = pd.Series(np.nan, index=df.index, dtype=float)
    hs_code = pd.to_numeric(df["hs_code"], errors="coerce")
    grade_if_neither = pd.to_numeric(df["grade_if_neither"], errors="coerce")
    college_years = pd.to_numeric(df["highest_year_college"], errors="coerce")
    received_degree = pd.to_numeric(df["received_degree"], errors="coerce")
    degree_type = pd.to_numeric(df["degree_type"], errors="coerce")

    years.loc[hs_code.eq(3)] = grade_if_neither.clip(lower=0, upper=11)
    years.loc[hs_code.isin([1, 2])] = 12
    years.loc[college_years.gt(0)] = 12 + college_years.clip(lower=0, upper=5)
    years.loc[received_degree.eq(1) & degree_type.eq(1)] = 14
    years.loc[received_degree.eq(1) & degree_type.eq(2)] = 16
    years.loc[received_degree.eq(1) & degree_type.isin([3, 5])] = 18
    years.loc[received_degree.eq(1) & degree_type.eq(4)] = 19
    years.loc[received_degree.eq(1) & degree_type.eq(6)] = 20
    years.loc[received_degree.eq(1) & degree_type.eq(97)] = np.maximum(
        years.loc[received_degree.eq(1) & degree_type.eq(97)].fillna(16),
        16,
    )
    return pd.to_numeric(years, errors="coerce")


def _education_level(years: pd.Series) -> pd.Series:
    result = pd.Series("unknown", index=years.index, dtype="string")
    y = pd.to_numeric(years, errors="coerce")
    result.loc[y.lt(12)] = "less_than_hs"
    result.loc[y.eq(12)] = "hs_diploma"
    result.loc[y.between(13, 15, inclusive="both")] = "some_college"
    result.loc[y.eq(16)] = "bachelors"
    result.loc[y.between(17, 18, inclusive="both")] = "masters"
    result.loc[y.ge(19)] = "professional_or_doctorate"
    return result


def _derive_recent_birth(youngest_child_birth_year: pd.Series, survey_year: int) -> pd.Series:
    year = pd.to_numeric(youngest_child_birth_year, errors="coerce")
    return year.between(survey_year - 1, survey_year, inclusive="both").astype(int)


def _derive_recent_marriage(most_recent_marriage_year: pd.Series, survey_year: int) -> pd.Series:
    year = pd.to_numeric(most_recent_marriage_year, errors="coerce")
    return year.between(survey_year - 1, survey_year, inclusive="both").astype(int)


def _annualize_salary(
    salary_amount: pd.Series,
    salary_per: pd.Series,
    pay_type: pd.Series,
    weeks_worked: pd.Series,
    usual_hours: pd.Series,
    hourly_rate: pd.Series,
) -> pd.Series:
    annual = pd.Series(np.nan, index=salary_amount.index, dtype=float)
    hourly_guess = pd.to_numeric(hourly_rate, errors="coerce") * pd.to_numeric(
        weeks_worked, errors="coerce"
    ) * pd.to_numeric(usual_hours, errors="coerce")
    annual.loc[pd.to_numeric(pay_type, errors="coerce").eq(3)] = hourly_guess

    multipliers = pd.to_numeric(salary_per, errors="coerce").map(
        {
            2: 52,   # weekly
            3: 26,   # biweekly
            4: 24,   # semimonthly
            5: 12,   # monthly
            6: 1,    # yearly
        }
    )
    annual = annual.fillna(pd.to_numeric(salary_amount, errors="coerce") * multipliers)
    annual = annual.fillna(hourly_guess)
    return annual


def _recode_psid_race_ethnicity(spanish_descent: pd.Series, race_mention_1: pd.Series) -> pd.Series:
    hisp = _clean_psid_code(spanish_descent, invalid_values={0, 9})
    race = _clean_psid_code(race_mention_1, invalid_values={0, 9})
    result = pd.Series("unknown", index=race.index, dtype="object")
    result.loc[race.eq(1)] = "white_non_hispanic"
    result.loc[race.eq(2)] = "black"
    result.loc[race.eq(3)] = "aian"
    result.loc[race.eq(4)] = "asian"
    result.loc[race.eq(5)] = "nhpi"
    result.loc[race.eq(7)] = "multiracial_other"
    result.loc[hisp.notna()] = "hispanic"
    return result


def _strip_psid_special_missing(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    rounded = series.round()
    integer_mask = series.notna() & np.isclose(series, rounded, equal_nan=False)
    integer_labels = pd.Series(pd.NA, index=series.index, dtype="string")
    integer_labels.loc[integer_mask] = rounded.loc[integer_mask].astype("Int64").astype(str)
    special_mask = integer_labels.str.fullmatch(r"[89]{2,}").fillna(False)
    series.loc[special_mask] = np.nan
    return series


def _clean_psid_positive_measure(values: pd.Series) -> pd.Series:
    series = _strip_psid_special_missing(values)
    series.loc[series <= 0] = np.nan
    return series


def _clean_psid_code(values: pd.Series, invalid_values: set[int] | None = None) -> pd.Series:
    series = _strip_psid_special_missing(values)
    if invalid_values:
        series.loc[series.isin(sorted(invalid_values))] = np.nan
    return series


def _clean_psid_year(values: pd.Series, survey_year: int) -> pd.Series:
    series = _strip_psid_special_missing(values)
    series.loc[series.lt(1900) | series.gt(survey_year)] = np.nan
    return series


def _clean_psid_age(values: pd.Series) -> pd.Series:
    series = _strip_psid_special_missing(values)
    series.loc[series.lt(0) | series.gt(120)] = np.nan
    return series


def _clean_psid_count(values: pd.Series) -> pd.Series:
    series = _strip_psid_special_missing(values)
    series.loc[series.lt(0)] = np.nan
    return series


def _clean_psid_weeks(values: pd.Series) -> pd.Series:
    series = _strip_psid_special_missing(values)
    series.loc[series.lt(0) | series.gt(52)] = np.nan
    return series


def _clean_psid_hours(values: pd.Series) -> pd.Series:
    series = _strip_psid_special_missing(values)
    series.loc[series.lt(0) | series.gt(168)] = np.nan
    return series


def _clean_psid_commute(values: pd.Series) -> pd.Series:
    series = _strip_psid_special_missing(values)
    series.loc[series.lt(0) | series.ge(900)] = np.nan
    return series


def _inflate_to_base_year(
    values: pd.Series,
    survey_year: int,
    cpi_index: dict[int, float] | None,
    base_year: int,
) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    if cpi_index and survey_year in cpi_index and base_year in cpi_index:
        return series * (cpi_index[base_year] / cpi_index[survey_year])
    return series


def _coerce_positive(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    series.loc[series <= 0] = np.nan
    return series


def _coerce_nonnegative(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    series.loc[series.lt(0)] = np.nan
    series.loc[series.ge(900)] = np.nan
    return series


def _self_employed_flag(class_of_worker: pd.Series) -> pd.Series:
    code = pd.to_numeric(class_of_worker, errors="coerce")
    result = pd.Series(pd.NA, index=code.index, dtype="Int64")
    result.loc[code.isin([1, 2])] = 0
    result.loc[code.eq(3)] = 1
    return result
