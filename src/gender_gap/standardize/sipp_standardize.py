"""Standardize official SIPP person-month style files into person_month_core.

This is a conservative first-pass standardizer for modern public-use SIPP files.
Variable names still vary by panel/module, so the implementation relies on
alias lists for the most common official fields and degrades gracefully when a
field is absent.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gender_gap.standardize.schema import PERSON_MONTH_CORE_COLUMNS

logger = logging.getLogger(__name__)


ALIASES = {
    "year": ["YEAR", "SURVEYYEAR", "RHCALYR", "REFYEAR"],
    "month": ["MONTHCODE", "MONTH", "RHCALMN", "REFMONTH"],
    "household_id": ["SSUID", "SU_ID", "SURID", "HHID"],
    "person_num": ["PNUM", "EPPPNUM", "P_NUM", "PERSONNUM"],
    "sex": ["ESEX", "SEX"],
    "age": ["TAGE", "AGE"],
    "education": ["EEDUC", "EDUC"],
    "employment_status": ["RMESR", "EMPSTAT"],
    "usual_hours": ["TJB1_JOBHRS1", "TJBHRS1", "EJBHRS1", "JBHRS1"],
    "actual_hours": ["TJB1_JOBHRS1", "AJBHRS1", "TJBHRS1", "EJBHRS1"],
    "hourly_pay": ["TJB1_HOURLY1", "TJB1_HRLYPAY", "EJB1_HRLYPAY", "TJBHRLY1", "HRRATE1"],
    "monthly_earnings": ["TJB1_MSUM", "EJB1_MSUM", "TPEARN", "TPMSUM1"],
    "annual_salary": ["TJB1_ANNSAL1", "EJB1_ANNSAL1", "ANNSAL1"],
    "paid_hourly_flag": ["EJB1_TYPPAY1", "EJB1_PAIDHOUR", "PAIDHOUR", "EJBPAIDH"],
    "overtime_flag": ["EJB1_OVERTIME", "OVERTIME", "EJBOT"],
    "occupation_code": ["TJB1_OCC", "EJB1_OCC", "OCC1", "TJOCC1"],
    "industry_code": ["TJB1_IND", "EJB1_IND", "IND1", "TJIND1"],
    "state_fips": ["TFIPSST", "STATEFIPS", "GESTFIPS", "STATE"],
    "weight": ["WPFINWGT", "FINWGT", "FINJTWGT"],
    "job2_earnings": ["TJB2_MSUM", "EJB2_MSUM", "TPMSUM2"],
}


def standardize_sipp(
    df: pd.DataFrame,
    cpi_index: dict[int, float] | None = None,
    base_year: int = 2024,
    survey_year: int | None = None,
) -> pd.DataFrame:
    """Transform a SIPP public-use file into person_month_core.

    Parameters
    ----------
    df : pd.DataFrame
        Raw SIPP public-use data.
    cpi_index : dict[int, float] | None
        Annual CPI-U index for deflation to a common base year.
    base_year : int
        Base year for deflation.
    survey_year : int | None
        Fallback year when no year column is present.
    """
    out = pd.DataFrame(index=df.index)

    year = _coerce_numeric(_series_from_aliases(df, "year"))
    if year is None and survey_year is not None:
        year = pd.Series(survey_year, index=df.index, dtype="Int64")
    month = _coerce_numeric(_series_from_aliases(df, "month"))
    if month is None:
        month = pd.Series(1, index=df.index, dtype="Int64")

    household_id = _series_from_aliases(df, "household_id")
    if household_id is None:
        household_id = pd.Series("sipp_household", index=df.index, dtype="string")
    person_num = _series_from_aliases(df, "person_num")
    if person_num is None:
        person_num = pd.Series(df.index.astype(str), index=df.index, dtype="string")

    out["person_id"] = household_id.astype("string") + "_" + person_num.astype("string")
    out["calendar_year"] = year
    out["month"] = month

    sex = _coerce_numeric(_series_from_aliases(df, "sex"))
    out["female"] = _indicator_from_code(sex, yes_codes={2})

    rmesr = _coerce_numeric(_series_from_aliases(df, "employment_status"))
    employed = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if rmesr is not None:
        employed = rmesr.between(1, 5).astype("Int64")
    out["employed"] = employed
    out["labor_force_status"] = _recode_sipp_labor_force(rmesr)

    usual_hours = _coerce_numeric(_series_from_aliases(df, "usual_hours"))
    actual_hours = _coerce_numeric(_series_from_aliases(df, "actual_hours"))
    out["usual_hours_week"] = usual_hours
    out["actual_hours_last_week"] = actual_hours

    hourly_pay = _coerce_numeric(_series_from_aliases(df, "hourly_pay"))
    monthly_earnings = _coerce_numeric(_series_from_aliases(df, "monthly_earnings"))
    annual_salary = _coerce_numeric(_series_from_aliases(df, "annual_salary"))

    weekly_earnings = pd.Series(np.nan, index=df.index, dtype=float)
    if monthly_earnings is not None:
        weekly_earnings = monthly_earnings / (52.0 / 12.0)
    if annual_salary is not None:
        weekly_from_annual = annual_salary / 52.0
        weekly_earnings = weekly_earnings.where(~weekly_earnings.isna(), weekly_from_annual)

    hourly_wage = pd.Series(np.nan, index=df.index, dtype=float)
    if hourly_pay is not None:
        hourly_wage = hourly_pay
    if usual_hours is not None:
        derived_hourly = weekly_earnings / usual_hours.replace(0, pd.NA)
        hourly_wage = hourly_wage.where(~hourly_wage.isna(), derived_hourly)

    deflator = _deflator(year, cpi_index, base_year)
    out["hourly_wage_real"] = hourly_wage * deflator
    out["weekly_earnings_real"] = weekly_earnings * deflator

    paid_hourly = _coerce_numeric(_series_from_aliases(df, "paid_hourly_flag"))
    if paid_hourly is not None:
        yes_codes = {1} if "EJB1_TYPPAY1" in df.columns else {1, 2}
        out["paid_hourly"] = _indicator_from_code(paid_hourly, yes_codes=yes_codes)
    else:
        out["paid_hourly"] = hourly_pay.notna().astype("Int64") if hourly_pay is not None else pd.Series(pd.NA, index=df.index, dtype="Int64")

    overtime_flag = _coerce_numeric(_series_from_aliases(df, "overtime_flag"))
    out["overtime_indicator"] = (
        _indicator_from_code(overtime_flag, yes_codes={1, 2})
        if overtime_flag is not None
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )

    second_job = _coerce_numeric(_series_from_aliases(df, "job2_earnings"))
    out["multiple_jobholder"] = (
        second_job.fillna(0).gt(0).astype("Int64")
        if second_job is not None
        else pd.Series(0, index=df.index, dtype="Int64")
    )

    out["occupation_code"] = _clean_string_codes(_series_from_aliases(df, "occupation_code"))
    out["industry_code"] = _clean_string_codes(_series_from_aliases(df, "industry_code"))
    out["state_fips"] = _coerce_numeric(_series_from_aliases(df, "state_fips"))

    weight = _coerce_numeric(_series_from_aliases(df, "weight"))
    out["person_weight"] = weight if weight is not None else 1.0

    for col in PERSON_MONTH_CORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[PERSON_MONTH_CORE_COLUMNS]


def _series_from_aliases(df: pd.DataFrame, key: str) -> pd.Series | None:
    for col in ALIASES[key]:
        if col in df.columns:
            return df[col]
    return None


def _clean_string_codes(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        cleaned = series.astype("string").str.strip()
        return cleaned.replace("", pd.NA)
    return series


def _coerce_numeric(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce")


def _indicator_from_code(series: pd.Series | None, yes_codes: set[int]) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, dtype="Int64")
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    valid = series.notna()
    out.loc[valid] = series.loc[valid].isin(yes_codes).astype("Int64")
    return out


def _deflator(
    year: pd.Series | None,
    cpi_index: dict[int, float] | None,
    base_year: int,
) -> pd.Series | float:
    if cpi_index is None or year is None:
        return 1.0
    return year.map(lambda y: cpi_index.get(base_year, 1.0) / cpi_index.get(int(y), 1.0) if pd.notna(y) else np.nan)


def _recode_sipp_labor_force(rmesr: pd.Series | None) -> pd.Series:
    if rmesr is None:
        return pd.Series(pd.NA, dtype="string")
    out = pd.Series("not_in_labor_force", index=rmesr.index, dtype="string")
    out.loc[rmesr.between(1, 5)] = "employed"
    out.loc[rmesr.isin([6, 7])] = "unemployed"
    out.loc[rmesr.isna()] = pd.NA
    return out
