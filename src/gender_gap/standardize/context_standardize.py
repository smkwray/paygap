"""Standardize contextual data sources into context_area_time schema.

Merges LAUS, QCEW, OEWS, and BEA RPP into a unified contextual table
keyed by geography + time.
"""

from __future__ import annotations

import logging

import pandas as pd

from gender_gap.standardize.schema import CONTEXT_AREA_TIME_COLUMNS

logger = logging.getLogger(__name__)


def _first_present(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for col in columns:
        if col in df.columns:
            return df[col]
    return pd.Series(pd.NA, index=df.index)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_string(series: pd.Series, width: int | None = None) -> pd.Series:
    out = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    if width is not None:
        out = out.str.zfill(width)
    return out


def standardize_laus(
    df: pd.DataFrame,
    geo_level: str = "state",
) -> pd.DataFrame:
    """Standardize LAUS data into context_area_time rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw LAUS data with columns: series_id, year, period, value.
    geo_level : str
        Geography level ('state', 'county', 'metro').
    """
    out = pd.DataFrame()
    out["geography_level"] = geo_level
    out["geography_key"] = df.get("area_code", df.get("series_id", pd.NA))
    out["calendar_year"] = df.get("year", pd.NA)
    out["local_unemployment_rate"] = df.get("unemployment_rate", pd.NA)
    out["local_labor_force"] = df.get("labor_force", pd.NA)
    out["local_industry_avg_weekly_wage"] = pd.NA
    out["local_industry_employment"] = pd.NA
    out["local_occupation_avg_wage"] = pd.NA
    out["local_price_parity"] = pd.NA

    return out[CONTEXT_AREA_TIME_COLUMNS]


def standardize_qcew(
    df: pd.DataFrame,
    geo_level: str = "county",
) -> pd.DataFrame:
    """Standardize QCEW data into context_area_time rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw QCEW annual averages with columns like area_fips,
        industry_code, annual_avg_wkly_wage, annual_avg_emplvl.
    geo_level : str
        Geography level.
    """
    area_fips = _to_string(_first_present(df, ["area_fips", "area_code"]), width=5)
    calendar_year = _to_numeric(_first_present(df, ["year"]))
    weekly_wage = _to_numeric(_first_present(df, ["annual_avg_wkly_wage"]))
    employment = _to_numeric(_first_present(df, ["annual_avg_emplvl"]))

    out = pd.DataFrame(index=df.index)
    out["geography_level"] = geo_level
    out["geography_key"] = area_fips
    out["calendar_year"] = calendar_year
    out["local_unemployment_rate"] = pd.NA
    out["local_labor_force"] = pd.NA
    out["local_industry_avg_weekly_wage"] = weekly_wage
    out["local_industry_employment"] = employment
    out["local_occupation_avg_wage"] = pd.NA
    out["local_price_parity"] = pd.NA

    return out[CONTEXT_AREA_TIME_COLUMNS]


def standardize_oews(
    df: pd.DataFrame,
    geo_level: str = "state",
) -> pd.DataFrame:
    """Standardize OEWS data into context_area_time rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OEWS data with columns like AREA, OCC_CODE, A_MEAN, A_MEDIAN.
    geo_level : str
        Geography level.
    """
    occ_code = _to_string(_first_present(df, ["OCC_CODE", "occ_code", "occ"]))
    if not occ_code.isna().all():
        keep = occ_code.isin(["00-0000", "000000"])
        if keep.any():
            df = df.loc[keep].copy()

    area_code = _to_string(_first_present(df, ["AREA", "area", "area_code"]))
    calendar_year = _to_numeric(_first_present(df, ["year", "YEAR"]))
    if calendar_year.isna().all():
        calendar_year = pd.Series(pd.NA, index=df.index)
    mean_wage = _to_numeric(_first_present(df, ["A_MEAN", "a_mean", "mean_wage"]))

    out = pd.DataFrame(index=df.index)
    out["geography_level"] = geo_level
    out["geography_key"] = area_code
    out["calendar_year"] = calendar_year
    out["local_unemployment_rate"] = pd.NA
    out["local_labor_force"] = pd.NA
    out["local_industry_avg_weekly_wage"] = pd.NA
    out["local_industry_employment"] = pd.NA
    out["local_occupation_avg_wage"] = mean_wage
    out["local_price_parity"] = pd.NA

    return out[CONTEXT_AREA_TIME_COLUMNS]


def standardize_bea_rpp(
    df: pd.DataFrame,
    geo_level: str = "state",
) -> pd.DataFrame:
    """Standardize BEA RPP data into context_area_time rows.

    Parameters
    ----------
    df : pd.DataFrame
        BEA RPP table with geography and RPP values.
    geo_level : str
        Geography level ('state' or 'metro').
    """
    geo_key = _to_string(_first_present(df, ["GeoFips", "geo_fips"]))
    calendar_year = _to_numeric(_first_present(df, ["year", "Year"]))
    rpp = _to_numeric(_first_present(df, ["rpp", "RPP"]))

    out = pd.DataFrame(index=df.index)
    out["geography_level"] = geo_level
    out["geography_key"] = geo_key
    out["calendar_year"] = calendar_year
    out["local_unemployment_rate"] = pd.NA
    out["local_labor_force"] = pd.NA
    out["local_industry_avg_weekly_wage"] = pd.NA
    out["local_industry_employment"] = pd.NA
    out["local_occupation_avg_wage"] = pd.NA
    out["local_price_parity"] = rpp

    return out[CONTEXT_AREA_TIME_COLUMNS]


def merge_context_tables(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Merge multiple context tables into one.

    Uses geography_level + geography_key + calendar_year as the merge key.
    Later tables fill in missing values from earlier tables.
    """
    if not tables:
        return pd.DataFrame(columns=CONTEXT_AREA_TIME_COLUMNS)

    merged = tables[0].copy()
    for t in tables[1:]:
        merged = pd.merge(
            merged,
            t,
            on=["geography_level", "geography_key", "calendar_year"],
            how="outer",
            suffixes=("", "_new"),
        )
        # Fill NAs from new table
        for col in CONTEXT_AREA_TIME_COLUMNS:
            new_col = f"{col}_new"
            if new_col in merged.columns:
                merged[col] = merged[col].fillna(merged[new_col])
                merged.drop(columns=[new_col], inplace=True)

    return merged[CONTEXT_AREA_TIME_COLUMNS]
