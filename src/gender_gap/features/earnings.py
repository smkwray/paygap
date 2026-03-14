"""Earnings feature engineering.

Handles inflation adjustment, hourly wage construction, winsorization,
and log transformations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gender_gap.settings import BASE_CURRENCY_YEAR

logger = logging.getLogger(__name__)


def compute_hourly_wage(
    earnings: pd.Series,
    hours: pd.Series,
    weeks: pd.Series | None = None,
    method: str = "annual",
) -> pd.Series:
    """Derive hourly wage from earnings and hours.

    Parameters
    ----------
    earnings : pd.Series
        Wage/salary income (annual or weekly depending on method).
    hours : pd.Series
        Usual hours per week.
    weeks : pd.Series | None
        Weeks worked (required for method='annual').
    method : str
        'annual' = earnings / (hours * weeks),
        'weekly' = earnings / hours.
    """
    safe_hours = hours.replace(0, np.nan)
    if method == "annual":
        if weeks is None:
            raise ValueError("weeks required for method='annual'")
        safe_weeks = weeks.replace(0, np.nan)
        return earnings / (safe_hours * safe_weeks)
    elif method == "weekly":
        return earnings / safe_hours
    else:
        raise ValueError(f"Unknown method: {method}")


def winsorize_wages(
    wages: pd.Series,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
    group_col: pd.Series | None = None,
) -> pd.Series:
    """Winsorize hourly wages at specified percentiles.

    Parameters
    ----------
    wages : pd.Series
        Hourly wage values.
    lower_pct : float
        Lower percentile (0-100).
    upper_pct : float
        Upper percentile (0-100).
    group_col : pd.Series | None
        Optional grouping variable (e.g., year) for within-group winsorization.
    """
    if group_col is not None:
        return wages.groupby(group_col).transform(
            lambda x: _clip_pct(x, lower_pct, upper_pct)
        )
    return _clip_pct(wages, lower_pct, upper_pct)


def _clip_pct(s: pd.Series, lower_pct: float, upper_pct: float) -> pd.Series:
    lo = np.nanpercentile(s.dropna(), lower_pct)
    hi = np.nanpercentile(s.dropna(), upper_pct)
    return s.clip(lower=lo, upper=hi)


def log_wage(wages: pd.Series) -> pd.Series:
    """Compute log(hourly_wage_real), setting non-positive to NaN."""
    w = wages.copy()
    w[w <= 0] = np.nan
    return np.log(w)


def deflate_series(
    values: pd.Series,
    source_years: pd.Series,
    cpi_index: dict[int, float],
    base_year: int = BASE_CURRENCY_YEAR,
) -> pd.Series:
    """Deflate a Series of nominal values to base-year dollars.

    Parameters
    ----------
    values : pd.Series
        Nominal dollar amounts.
    source_years : pd.Series
        Year for each observation.
    cpi_index : dict[int, float]
        Year -> CPI-U annual average.
    base_year : int
        Target base year.
    """
    if base_year not in cpi_index:
        raise ValueError(f"base_year {base_year} not in cpi_index")
    base_cpi = cpi_index[base_year]
    deflator = source_years.map(lambda y: base_cpi / cpi_index.get(y, np.nan))
    return values * deflator
