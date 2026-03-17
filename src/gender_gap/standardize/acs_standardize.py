"""Standardize ACS PUMS into person_year_core schema."""

from __future__ import annotations

import logging

import pandas as pd

from gender_gap.standardize.schema import PERSON_YEAR_CORE_COLUMNS

logger = logging.getLogger(__name__)

# ACS PUMS commute mode mapping (JWTRNS values)
COMMUTE_MODE_MAP = {
    1: "car_truck_van_alone",
    2: "car_truck_van_carpool",
    3: "bus",
    4: "streetcar_trolley",
    5: "subway_elevated",
    6: "railroad",
    7: "ferryboat",
    8: "taxicab",
    9: "motorcycle",
    10: "bicycle",
    11: "walked",
    12: "work_from_home",
}


def standardize_acs(
    df: pd.DataFrame,
    survey_year: int | None = None,
    adj_factor_col: str = "ADJINC",
    keep_replicate_weights: bool = False,
) -> pd.DataFrame:
    """Transform raw ACS PUMS person file into person_year_core schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ACS PUMS person-level data.
    survey_year : int
        The ACS survey year.
    adj_factor_col : str
        Column containing the ACS income adjustment factor (divided by 1_000_000).
    keep_replicate_weights : bool
        If true, append `PWGTP1`-`PWGTP80` columns after the core schema.

    Returns
    -------
    pd.DataFrame
        Standardized person_year_core table.
    """
    out = pd.DataFrame(index=df.index)
    age = _numeric_series(df.get("AGEP"), index=df.index)
    sex = _numeric_series(df.get("SEX"), index=df.index)
    schl = _numeric_series(df.get("SCHL"), index=df.index)
    mar = _numeric_series(df.get("MAR"), index=df.index)
    cow = _numeric_series(df.get("COW"), index=df.index)
    hisp = _numeric_series(df.get("HISP"), index=df.index)
    rac1p = _numeric_series(df.get("RAC1P"), index=df.index)
    fer = _numeric_series(df.get("FER"), index=df.index)
    marhm = _numeric_series(df.get("MARHM"), index=df.index)
    cplt = _numeric_series(df.get("CPLT"), index=df.index)
    partner = _numeric_series(df.get("PARTNER"), index=df.index)
    relshipp = _numeric_series(df.get("RELSHIPP", df.get("RELP")), index=df.index)
    noc = _numeric_series(df.get("NOC"), index=df.index)
    paoc = _numeric_series(df.get("PAOC"), index=df.index)
    jwtrns = _numeric_series(df.get("JWTRNS", df.get("JWTR")), index=df.index)
    if survey_year is None:
        survey_year = _infer_survey_year(df)
    if survey_year is None:
        raise ValueError("survey_year is required when the ACS frame does not include YEAR")

    # Keys
    out["person_id"] = df["SERIALNO"].astype(str) + "_" + df["SPORDER"].astype(str)
    out["household_id"] = df["SERIALNO"].astype(str)
    out["data_source"] = "ACS"
    out["survey_year"] = survey_year
    out["calendar_year"] = survey_year
    out["acs_serialno"] = df["SERIALNO"].astype(str)
    out["acs_sporder"] = _numeric_series(df.get("SPORDER"), index=df.index)

    # Demographics
    out["female"] = (sex == 2).astype(int)
    out["age"] = age
    out["age_sq"] = age ** 2

    out["race_ethnicity"] = _recode_race(hisp, rac1p)
    out["education_level"] = _recode_education(schl)
    out["marital_status"] = _recode_marital(mar)

    # Family
    # `NOC` and `PAOC` are ACS PUMS person-level summaries for own children.
    out["fer"] = fer
    out["marhm"] = marhm
    out["cplt"] = cplt
    out["partner"] = partner
    out["relshipp"] = relshipp
    out["paoc"] = paoc
    out["noc"] = noc
    out["number_children"] = _recode_number_children(df)
    out["children_under_5"] = _recode_children_under_5(df)
    out["recent_birth"] = (fer == 1).astype(int)
    out["recent_marriage"] = _recode_recent_marriage(marhm)
    out["has_own_child"] = _recode_has_own_child(noc, paoc)
    out["own_child_under6"] = paoc.isin([1, 3]).astype(int)
    out["own_child_6_17_only"] = (paoc == 2).astype(int)
    out["same_sex_couple_household"] = cplt.isin([2, 4]).astype(int)
    out["opposite_sex_couple_household"] = cplt.isin([1, 3]).astype(int)
    out["couple_type"] = _recode_couple_type(out)
    out["reproductive_stage"] = _recode_reproductive_stage(out)
    out["age_fertility_band"] = _age_fertility_band(age)
    out["older_low_fertility_placebo"] = age.between(45, 54, inclusive="both").astype(int)

    # Job
    out["occupation_code"] = df.get("OCCP", df.get("SOCP", pd.NA))
    out["industry_code"] = df.get("INDP", pd.NA)
    out["class_of_worker"] = cow
    out["self_employed"] = cow.isin([6, 7]).astype(int)

    # Schedule
    out["weeks_worked"] = df.get("WKWN", df.get("WKW", pd.NA))
    out["usual_hours_week"] = df.get("WKHP", pd.NA)
    out["annual_hours"] = out["usual_hours_week"] * out["weeks_worked"]

    # Commute
    out["work_from_home"] = (jwtrns == 12).astype(int) if jwtrns is not None else 0
    out["commute_minutes_one_way"] = df.get("JWMNP", pd.NA)
    out["commute_mode"] = jwtrns.map(COMMUTE_MODE_MAP) if jwtrns is not None else pd.NA

    # Geography
    out["state_fips"] = df.get("ST", df.get("STATE", pd.NA))
    out["residence_puma"] = df.get("PUMA", pd.NA)
    out["place_of_work_state"] = df.get("POWSP", pd.NA)
    out["place_of_work_puma"] = df.get("POWPUMA", pd.NA)

    # Earnings (real dollars using ACS adjustment factor)
    adj = df.get(adj_factor_col, 1_000_000) / 1_000_000
    wagp = df.get("WAGP", pd.Series(0, index=df.index, dtype=float)) * adj
    pernp = df.get("PERNP", pd.Series(0, index=df.index, dtype=float)) * adj

    out["wage_salary_income_real"] = wagp
    out["annual_earnings_real"] = pernp
    # Hourly wage = real wage-salary / annual hours
    annual_hours = out["annual_hours"].replace(0, pd.NA)
    out["hourly_wage_real"] = wagp / annual_hours

    # Weight
    out["person_weight"] = df.get("PWGTP", 1)

    # Ensure column order matches schema
    for col in PERSON_YEAR_CORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    result = out[PERSON_YEAR_CORE_COLUMNS]
    if keep_replicate_weights:
        repweight_cols = [
            col for col in df.columns
            if col.startswith("PWGTP") and col != "PWGTP"
        ]
        if repweight_cols:
            result = pd.concat([result, df[repweight_cols].copy()], axis=1)
    return result


def _numeric_series(values, index) -> pd.Series:
    """Coerce an ACS raw column to numeric while preserving row alignment."""
    if values is None:
        return pd.Series(pd.NA, index=index, dtype="Float64")
    return pd.to_numeric(values, errors="coerce")


def _recode_race(hisp: pd.Series, rac1p: pd.Series) -> pd.Series:
    """Recode ACS HISP + RAC1P into mutually exclusive race/ethnicity."""
    result = pd.Series("other", index=hisp.index)
    result[rac1p == 1] = "white_non_hispanic"
    result[rac1p == 2] = "black"
    result[rac1p.isin([3, 4, 5])] = "aian"
    result[rac1p == 6] = "asian"
    result[rac1p == 7] = "nhpi"
    result[rac1p.isin([8, 9])] = "multiracial_other"
    # Hispanic overrides
    result[hisp > 1] = "hispanic"
    return result


def _recode_education(schl: pd.Series) -> pd.Series:
    """Recode ACS SCHL into ordered education levels."""
    result = pd.Series("unknown", index=schl.index)
    result[schl <= 15] = "less_than_hs"
    result[(schl >= 16) & (schl <= 17)] = "hs_diploma"
    result[(schl >= 18) & (schl <= 19)] = "some_college"
    result[schl == 20] = "associates"
    result[schl == 21] = "bachelors"
    result[schl == 22] = "masters"
    result[schl == 23] = "professional"
    result[schl == 24] = "doctorate"
    return result


def _recode_marital(mar: pd.Series) -> pd.Series:
    """Recode ACS MAR into standard categories."""
    mapping = {
        1: "married",
        2: "widowed",
        3: "divorced",
        4: "separated",
        5: "never_married",
    }
    return mar.map(mapping).fillna("unknown")


def _recode_number_children(df: pd.DataFrame) -> pd.Series:
    """Use ACS person-level number-of-own-children when available."""
    noc = pd.to_numeric(df.get("NOC", 0), errors="coerce").fillna(0)
    return noc.clip(lower=0).astype(int)


def _recode_children_under_5(df: pd.DataFrame) -> pd.Series:
    """Convert ACS `PAOC` presence/age codes into an under-6 indicator.

    ACS exposes whether own children are present and whether any are under 6,
    but not the exact count under age 6 in the person API extract.
    The standardized field is therefore a binary indicator.
    """
    paoc = pd.to_numeric(df.get("PAOC", 0), errors="coerce").fillna(0)
    return paoc.isin([1, 3]).astype(int)


def _recode_recent_marriage(marhm: pd.Series) -> pd.Series:
    """Convert ACS MARHM into a married-within-12-months indicator."""
    return marhm.between(1, 12, inclusive="both").astype(int)


def _recode_has_own_child(noc: pd.Series, paoc: pd.Series) -> pd.Series:
    """Prefer NOC with PAOC as a fallback presence check."""
    return ((noc.fillna(0) > 0) | paoc.isin([1, 2, 3])).astype(int)


def _recode_couple_type(df: pd.DataFrame) -> pd.Series:
    """Collapse same/opposite-sex couple structure into one label."""
    result = pd.Series("unpartnered", index=df.index, dtype="string")
    result.loc[df["opposite_sex_couple_household"] == 1] = "opposite_sex"
    result.loc[df["same_sex_couple_household"] == 1] = "same_sex"
    return result


def _recode_reproductive_stage(df: pd.DataFrame) -> pd.Series:
    """Build a mutually exclusive reproductive-stage label."""
    result = pd.Series("childless_unpartnered", index=df.index, dtype="string")

    childless = df["has_own_child"].fillna(0) == 0
    partnered = (
        df["same_sex_couple_household"].fillna(0).astype(int)
        | df["opposite_sex_couple_household"].fillna(0).astype(int)
        | df["partner"].fillna(0).gt(0).astype(int)
    ).astype(bool)

    result.loc[childless & partnered] = "childless_other_partnered"
    result.loc[childless & (df["recent_marriage"].fillna(0) == 1)] = "childless_recently_married"
    result.loc[df["own_child_6_17_only"].fillna(0) == 1] = "mother_6_17_only"
    result.loc[df["own_child_under6"].fillna(0) == 1] = "mother_under6"
    result.loc[df["has_own_child"].fillna(0).eq(1) & ~df["own_child_under6"].fillna(0).eq(1) & ~df["own_child_6_17_only"].fillna(0).eq(1)] = "mother_mixed_or_other"
    result.loc[df["recent_birth"].fillna(0) == 1] = "recent_birth"
    return result


def _age_fertility_band(age: pd.Series) -> pd.Series:
    """Bucket age into the seeded fertility-analysis bands."""
    result = pd.Series("outside_range", index=age.index, dtype="string")
    bands = [
        (25, 29, "25_29"),
        (30, 34, "30_34"),
        (35, 39, "35_39"),
        (40, 44, "40_44"),
        (45, 49, "45_49"),
        (50, 54, "50_54"),
    ]
    for lo, hi, label in bands:
        result.loc[age.between(lo, hi, inclusive="both")] = label
    return result


def _infer_survey_year(df: pd.DataFrame) -> int | None:
    """Infer the ACS survey year when a YEAR column is present."""
    if "YEAR" not in df.columns:
        return None
    values = pd.to_numeric(df["YEAR"], errors="coerce").dropna().astype(int)
    if values.empty:
        return None
    return int(values.iloc[0])
