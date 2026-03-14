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
    survey_year: int,
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
    out = pd.DataFrame()
    age = _numeric_series(df.get("AGEP"), index=df.index)
    sex = _numeric_series(df.get("SEX"), index=df.index)
    schl = _numeric_series(df.get("SCHL"), index=df.index)
    mar = _numeric_series(df.get("MAR"), index=df.index)
    cow = _numeric_series(df.get("COW"), index=df.index)
    hisp = _numeric_series(df.get("HISP"), index=df.index)
    rac1p = _numeric_series(df.get("RAC1P"), index=df.index)
    jwtrns = _numeric_series(df.get("JWTRNS", df.get("JWTR")), index=df.index)

    # Keys
    out["person_id"] = df["SERIALNO"].astype(str) + "_" + df["SPORDER"].astype(str)
    out["household_id"] = df["SERIALNO"].astype(str)
    out["data_source"] = "ACS"
    out["survey_year"] = survey_year
    out["calendar_year"] = survey_year

    # Demographics
    out["female"] = (sex == 2).astype(int)
    out["age"] = age
    out["age_sq"] = age ** 2

    out["race_ethnicity"] = _recode_race(hisp, rac1p)
    out["education_level"] = _recode_education(schl)
    out["marital_status"] = _recode_marital(mar)

    # Family
    # `NOC` and `PAOC` are ACS PUMS person-level summaries for own children.
    out["number_children"] = _recode_number_children(df)
    out["children_under_5"] = _recode_children_under_5(df)

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
    out["state_fips"] = df.get("ST", pd.NA)
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
