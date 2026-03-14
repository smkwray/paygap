"""Sample filter definitions.

Implements the sample restrictions described in the project brief:
- S1: prime-age wage-and-salary workers (25-54)
- S2: all employed workers (18-64)
- S3: parenthood strata
- S4: commute-rich workers
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def filter_prime_age_wage_salary(
    df: pd.DataFrame,
    age_min: int = 25,
    age_max: int = 54,
) -> pd.DataFrame:
    """S1: Prime-age wage-and-salary workers.

    Filters:
    - age 25-54
    - not self-employed
    - positive hours
    - positive wage/earnings
    """
    mask = (
        (df["age"] >= age_min)
        & (df["age"] <= age_max)
        & (df["self_employed"] == 0)
        & (df["usual_hours_week"] > 0)
        & (df["hourly_wage_real"] > 0)
    )
    n_before = len(df)
    result = df[mask].copy()
    logger.info(
        "S1 prime-age wage-salary: %d -> %d rows (%.1f%% kept)",
        n_before, len(result), 100 * len(result) / max(n_before, 1),
    )
    return result


def filter_all_employed(
    df: pd.DataFrame,
    age_min: int = 18,
    age_max: int = 64,
) -> pd.DataFrame:
    """S2: All employed workers 18-64."""
    mask = (
        (df["age"] >= age_min)
        & (df["age"] <= age_max)
    )
    # For person_year_core, check weeks/hours; for person_month, check employed
    if "weeks_worked" in df.columns:
        mask = mask & (df["weeks_worked"] > 0) & (df["usual_hours_week"] > 0)
    elif "employed" in df.columns:
        mask = mask & (df["employed"] == 1)

    n_before = len(df)
    result = df[mask].copy()
    logger.info(
        "S2 all employed: %d -> %d rows (%.1f%% kept)",
        n_before, len(result), 100 * len(result) / max(n_before, 1),
    )
    return result


def filter_commute_rich(
    df: pd.DataFrame,
    exclude_wfh: bool = True,
) -> pd.DataFrame:
    """S4: Workers with non-missing commute data.

    If exclude_wfh=True, drops work-from-home workers.
    """
    mask = df["commute_minutes_one_way"].notna()
    if exclude_wfh:
        mask = mask & (df["work_from_home"] == 0)

    n_before = len(df)
    result = df[mask].copy()
    logger.info(
        "S4 commute-rich (exclude_wfh=%s): %d -> %d rows",
        exclude_wfh, n_before, len(result),
    )
    return result


def drop_outlier_wages(
    df: pd.DataFrame,
    wage_col: str = "hourly_wage_real",
    min_wage: float = 2.0,
    max_wage: float = 500.0,
) -> pd.DataFrame:
    """Drop extreme wage outliers before analysis.

    Default: drop wages below $2/hr or above $500/hr.
    Winsorization is handled separately in earnings.py.
    """
    mask = (df[wage_col] >= min_wage) & (df[wage_col] <= max_wage)
    n_before = len(df)
    result = df[mask].copy()
    dropped = n_before - len(result)
    if dropped > 0:
        logger.info(
            "Dropped %d rows with %s outside [%.1f, %.1f]",
            dropped, wage_col, min_wage, max_wage,
        )
    return result
