"""Standardize CPS ORG / Basic Monthly into person_month_core schema.

Supports both IPUMS CPS harmonized variable names and official raw variable names.
The IPUMS path is the default and recommended approach.
"""

from __future__ import annotations

import logging

import pandas as pd

from gender_gap.standardize.schema import PERSON_MONTH_CORE_COLUMNS

logger = logging.getLogger(__name__)


def standardize_cps_ipums(
    df: pd.DataFrame,
    cpi_index: dict[int, float] | None = None,
    base_year: int = 2024,
) -> pd.DataFrame:
    """Transform IPUMS CPS extract into person_month_core schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw IPUMS CPS data with harmonized variable names.
    cpi_index : dict[int, float] | None
        Year -> CPI-U annual average. If None, earnings are kept nominal.
    base_year : int
        Target year for real-dollar deflation.

    Returns
    -------
    pd.DataFrame
        Standardized person_month_core table (ORG wage-eligible rows only).
    """
    out = pd.DataFrame()

    # Keys
    out["person_id"] = (
        df["YEAR"].astype(str) + "_"
        + df["SERIAL"].astype(str) + "_"
        + df["PERNUM"].astype(str)
    )
    out["calendar_year"] = df["YEAR"]
    out["month"] = df["MONTH"]

    # Demographics
    out["female"] = (df["SEX"] == 2).astype(int)

    # Employment status (IPUMS EMPSTAT: 10-12 = employed)
    empstat = df.get("EMPSTAT", pd.Series(dtype="Int64"))
    out["employed"] = empstat.between(10, 12).astype(int) if empstat is not None else pd.NA

    # Labor force status
    out["labor_force_status"] = _recode_lf_status(empstat)

    # Hours
    out["usual_hours_week"] = df.get("UHRSWORKORG", df.get("UHRSWORKT", pd.NA))
    out["actual_hours_last_week"] = df.get("AHRSWORKT", pd.NA)

    # Pay basis
    paidhour = df.get("PAIDHOUR", pd.Series(dtype="Int64"))
    out["paid_hourly"] = (paidhour == 2).astype(int) if paidhour is not None else pd.NA

    # Earnings
    hourwage = df.get("HOURWAGE", pd.Series(dtype=float))
    earnweek = df.get("EARNWEEK", pd.Series(dtype=float))
    uhrs = out["usual_hours_week"]

    # Derive hourly wage: direct if paid hourly, else weekly/usual hours
    hourly = hourwage.copy()
    derive_mask = hourly.isna() | (hourly <= 0)
    derived = earnweek / uhrs.replace(0, pd.NA)
    hourly = hourly.where(~derive_mask, derived)

    # Deflate to real dollars
    if cpi_index is not None:
        deflator = df["YEAR"].map(
            lambda y: cpi_index.get(base_year, 1.0) / cpi_index.get(y, 1.0)
        )
    else:
        deflator = 1.0

    out["hourly_wage_real"] = hourly * deflator
    out["weekly_earnings_real"] = earnweek * deflator

    # Overtime
    otpay = df.get("OTPAY", pd.Series(dtype="Int64"))
    out["overtime_indicator"] = (otpay == 2).astype(int) if otpay is not None else pd.NA

    # Multiple jobholder
    multjob = df.get("MULTJOB", df.get("PEMJOT", pd.Series(dtype="Int64")))
    out["multiple_jobholder"] = (multjob == 2).astype(int) if multjob is not None else pd.NA

    # Occupation and industry
    out["occupation_code"] = df.get("OCC", df.get("OCC2010", pd.NA))
    out["industry_code"] = df.get("IND", df.get("IND1990", pd.NA))

    # Geography
    out["state_fips"] = df.get("STATEFIP", pd.NA)

    # Weight — use EARNWT for ORG analyses, fall back to WTFINL
    out["person_weight"] = df.get("EARNWT", df.get("WTFINL", 1.0))

    # Ensure all schema columns present
    for col in PERSON_MONTH_CORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[PERSON_MONTH_CORE_COLUMNS]


def standardize_cps_official(
    df: pd.DataFrame,
    cpi_index: dict[int, float] | None = None,
    base_year: int = 2024,
) -> pd.DataFrame:
    """Transform official Census/BLS CPS raw variables into person_month_core.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CPS Basic Monthly public-use file.
    cpi_index : dict[int, float] | None
        Year -> CPI-U annual average for deflation.
    base_year : int
        Target year for real-dollar deflation.

    Returns
    -------
    pd.DataFrame
        Standardized person_month_core table.
    """
    out = pd.DataFrame()

    year_col = _get_col(df, ["HRYEAR4", "YEAR"])
    month_col = _get_col(df, ["HRMONTH", "MONTH"])

    out["person_id"] = (
        df[year_col].astype(str) + "_"
        + df[month_col].astype(str) + "_"
        + df.get("HRHHID", df.index).astype(str) + "_"
        + df.get("PULINENO", df.index).astype(str)
    )
    out["calendar_year"] = df[year_col]
    out["month"] = df[month_col]

    # Demographics
    pesex = df.get("PESEX", pd.Series(dtype="Int64"))
    out["female"] = (pesex == 2).astype(int) if pesex is not None else pd.NA

    # Employment
    pemlr = df.get("PEMLR", pd.Series(dtype="Int64"))
    out["employed"] = pemlr.isin([1, 2]).astype(int) if pemlr is not None else pd.NA
    out["labor_force_status"] = _recode_lf_status_official(pemlr)

    # Hours
    out["usual_hours_week"] = df.get("PEHRUSL1", df.get("PEHRUSLT", pd.NA))
    out["actual_hours_last_week"] = df.get("PEHRACT1", df.get("PEHRACTT", pd.NA))

    # Pay basis
    peernhro = df.get("PEERNHRO", pd.Series(dtype="Int64"))
    out["paid_hourly"] = (peernhro == 1).astype(int) if peernhro is not None else pd.NA

    # Earnings
    hourwage_raw = df.get("PTERNHLY", pd.Series(dtype=float))
    earnweek_raw = df.get("PTERNWA", pd.Series(dtype=float))
    uhrs = out["usual_hours_week"]

    # CPS stores cents (implied 2 decimal): divide by 100
    hourwage = hourwage_raw / 100.0 if hourwage_raw is not None else pd.NA
    earnweek = earnweek_raw / 100.0 if earnweek_raw is not None else pd.NA

    if isinstance(hourwage, pd.Series):
        hourly = hourwage.copy()
    else:
        hourly = pd.Series(pd.NA, index=df.index)
    derive_mask = hourly.isna() | (hourly <= 0)
    derived = earnweek / uhrs.replace(0, pd.NA) if isinstance(earnweek, pd.Series) else pd.NA
    hourly = hourly.where(~derive_mask, derived)

    if cpi_index is not None:
        deflator = df[year_col].map(
            lambda y: cpi_index.get(base_year, 1.0) / cpi_index.get(y, 1.0)
        )
    else:
        deflator = 1.0

    out["hourly_wage_real"] = hourly * deflator
    out["weekly_earnings_real"] = earnweek * deflator if isinstance(earnweek, pd.Series) else pd.NA

    # Overtime
    peernuot = df.get("PEERNUOT", pd.Series(dtype="Int64"))
    out["overtime_indicator"] = (peernuot == 1).astype(int) if peernuot is not None else pd.NA

    # Multiple jobholder
    pemjot = df.get("PEMJOT", pd.Series(dtype="Int64"))
    out["multiple_jobholder"] = (pemjot == 1).astype(int) if pemjot is not None else pd.NA

    # Occupation and industry
    out["occupation_code"] = df.get("PRMJOCC1", pd.NA)
    out["industry_code"] = df.get("PRMJIND1", pd.NA)

    # Geography
    out["state_fips"] = df.get("GESTFIPS", df.get("GESTCEN", pd.NA))

    # Weight
    out["person_weight"] = df.get("PWSSWGT", df.get("PWORWGT", 1.0))

    for col in PERSON_MONTH_CORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[PERSON_MONTH_CORE_COLUMNS]


def _recode_lf_status(empstat: pd.Series) -> pd.Series:
    """Recode IPUMS EMPSTAT to labor force status string."""
    result = pd.Series("not_in_labor_force", index=empstat.index)
    result[empstat.between(10, 12)] = "employed"
    result[empstat.between(20, 22)] = "unemployed"
    return result


def _recode_lf_status_official(pemlr: pd.Series) -> pd.Series:
    """Recode official CPS PEMLR to labor force status string."""
    result = pd.Series("not_in_labor_force", index=pemlr.index)
    result[pemlr.isin([1, 2])] = "employed"
    result[pemlr.isin([3, 4])] = "unemployed"
    return result


def _get_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name found in df from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in DataFrame columns")
