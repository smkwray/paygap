"""Standardize ATUS data into person_day_timeuse schema.

ATUS is a mechanism module — it measures daily time allocation, not
annual earnings. Do not naively merge ATUS onto ACS/CPS person-year rows.

Supports both BLS raw ATUS files and IPUMS ATUS-X extracts.
"""

from __future__ import annotations

import logging

import pandas as pd

from gender_gap.standardize.schema import PERSON_DAY_TIMEUSE_COLUMNS

logger = logging.getLogger(__name__)

# BLS ATUS activity code prefixes for key categories
# See https://www.bls.gov/tus/lexicons.htm
PAID_WORK_PREFIX = "0501"       # Working at main/other job
WORK_AT_HOME_LOCATION = 1       # Home location code
COMMUTE_TRAVEL_PREFIX = "1805"  # Travel related to work
HOUSEWORK_PREFIX = "0201"       # Housework
CHILDCARE_PREFIX = "0301"       # Caring for HH children
ELDERCARE_PREFIX = "0302"       # Caring for HH adults


def standardize_atus_summary(
    df: pd.DataFrame,
    year_col: str = "TUYEAR",
) -> pd.DataFrame:
    """Transform ATUS activity summary file into person_day_timeuse schema.

    The summary file has pre-aggregated time variables (minutes per day
    in each activity category) for each respondent-day.

    Parameters
    ----------
    df : pd.DataFrame
        ATUS activity summary (atussum) with time variables.
    year_col : str
        Column containing the survey year.
    """
    out = pd.DataFrame()

    # Identifiers
    out["person_id"] = df.get("TUCASEID", df.index).astype(str)
    out["calendar_year"] = df.get(year_col, df.get("TUYEAR", pd.NA))
    out["diary_date"] = df.get("TUDIESSION", df.get("TUDIARYDATE", pd.NA))

    # Demographics
    tesex = df.get("TESEX", df.get("SEX", pd.Series(dtype="Int64")))
    out["female"] = (tesex == 2).astype(int) if tesex is not None else pd.NA

    # Employment
    telfs = df.get("TELFS", pd.Series(dtype="Int64"))
    out["employed"] = telfs.isin([1, 2]).astype(int) if telfs is not None else pd.NA

    # Time use variables (in minutes)
    # BLS summary file variable naming: t{6-digit activity code}
    _work_codes = [
        "t050101", "t050102", "t050103", "t050189",
        "t050201", "t050202", "t050203", "t050289",
    ]
    out["minutes_paid_work_diary"] = _sum_cols(df, _work_codes)
    out["minutes_work_at_home_diary"] = df.get("t050103", 0)
    _commute_codes = ["t180501", "t180502", "t180589"]
    out["minutes_commute_related_travel"] = _sum_cols(df, _commute_codes)
    _hw_codes = ["t020101", "t020102", "t020103", "t020104", "t020199"]
    out["minutes_housework"] = _sum_cols(df, _hw_codes)
    _cc_codes = [
        "t030101", "t030102", "t030103", "t030104",
        "t030105", "t030108", "t030109", "t030110", "t030199",
    ]
    out["minutes_childcare"] = _sum_cols(df, _cc_codes)
    _ec_codes = ["t030201", "t030202", "t030203", "t030204", "t030299"]
    out["minutes_eldercare"] = _sum_cols(df, _ec_codes)
    out["minutes_with_children"] = pd.NA  # Requires who-file linkage

    # Weight
    wt = df.get("TUFINLWGT", df.get("TU20FWGT", df.get("TUFNWGTP", 1.0)))
    out["person_weight"] = wt

    for col in PERSON_DAY_TIMEUSE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[PERSON_DAY_TIMEUSE_COLUMNS]


def standardize_atus_ipums(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Transform IPUMS ATUS-X extract into person_day_timeuse schema.

    IPUMS ATUS-X provides pre-harmonized time-use summary variables
    with BLS_ prefixes.

    Parameters
    ----------
    df : pd.DataFrame
        IPUMS ATUS-X extract.
    """
    out = pd.DataFrame()

    out["person_id"] = df.get("CASEID", df.index).astype(str)
    out["calendar_year"] = df.get("YEAR", pd.NA)
    out["diary_date"] = df.get("DATE", df.get("DAY", pd.NA))

    # Demographics
    sex = df.get("SEX", pd.Series(dtype="Int64"))
    out["female"] = (sex == 2).astype(int) if sex is not None else pd.NA

    empstat = df.get("EMPSTAT", pd.Series(dtype="Int64"))
    out["employed"] = empstat.isin([1, 2]).astype(int) if empstat is not None else pd.NA

    # IPUMS BLS time-use summary variables (minutes)
    out["minutes_paid_work_diary"] = df.get("BLS_WORK", 0)
    out["minutes_work_at_home_diary"] = df.get("BLS_WORK_HOME", 0)
    out["minutes_commute_related_travel"] = df.get("BLS_COMM", df.get("BLS_TRAV_WORK", 0))
    out["minutes_housework"] = df.get("BLS_HHACT", 0)
    out["minutes_childcare"] = df.get("BLS_CAREHH", df.get("BLS_CARECH", 0))
    out["minutes_eldercare"] = df.get("BLS_CAREAD", 0)
    out["minutes_with_children"] = pd.NA

    wt = df.get("WT06", df.get("WGTP", 1.0))
    out["person_weight"] = wt

    for col in PERSON_DAY_TIMEUSE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[PERSON_DAY_TIMEUSE_COLUMNS]


def _sum_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Sum available columns, treating missing columns as 0."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(0, index=df.index)
    return df[present].fillna(0).sum(axis=1)
