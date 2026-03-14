"""Descriptive weighted gap tables.

Computes weighted mean and median wage gaps by subgroups.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from gender_gap.utils.weights import (
    replicate_weight_columns,
    sdr_summary,
    weighted_mean,
    weighted_quantile,
)

logger = logging.getLogger(__name__)


def raw_gap(
    df: pd.DataFrame,
    outcome: str = "hourly_wage_real",
    weight: str = "person_weight",
) -> dict:
    """Compute the national weighted raw gap.

    Returns
    -------
    dict with keys: male_mean, female_mean, gap_dollars, gap_pct, n_male, n_female
    """
    male = df[df["female"] == 0]
    female = df[df["female"] == 1]

    m_mean = weighted_mean(male[outcome], male[weight])
    f_mean = weighted_mean(female[outcome], female[weight])
    gap_dollars = m_mean - f_mean
    gap_pct = gap_dollars / m_mean * 100 if m_mean != 0 else float("nan")

    return {
        "male_mean": m_mean,
        "female_mean": f_mean,
        "gap_dollars": gap_dollars,
        "gap_pct": gap_pct,
        "n_male": len(male),
        "n_female": len(female),
    }


def gap_by_subgroup(
    df: pd.DataFrame,
    group_col: str,
    outcome: str = "hourly_wage_real",
    weight: str = "person_weight",
) -> pd.DataFrame:
    """Compute weighted gap by subgroup.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'female', outcome, and weight columns.
    group_col : str
        Column to group by (e.g., 'education_level', 'occupation_code').

    Returns
    -------
    pd.DataFrame with columns: group, male_mean, female_mean, gap_dollars, gap_pct, n
    """
    rows = []
    for group_val, gdf in df.groupby(group_col, observed=True):
        male = gdf[gdf["female"] == 0]
        female = gdf[gdf["female"] == 1]
        if len(male) == 0 or len(female) == 0:
            continue
        m_mean = weighted_mean(male[outcome], male[weight])
        f_mean = weighted_mean(female[outcome], female[weight])
        gap = m_mean - f_mean
        gap_pct = gap / m_mean * 100 if m_mean != 0 else float("nan")
        rows.append({
            "group": group_val,
            "male_mean": m_mean,
            "female_mean": f_mean,
            "gap_dollars": gap,
            "gap_pct": gap_pct,
            "n": len(gdf),
        })

    return pd.DataFrame(rows)


def gap_table(
    df: pd.DataFrame,
    group_cols: list[str],
    outcome: str = "hourly_wage_real",
    weight: str = "person_weight",
) -> dict[str, pd.DataFrame]:
    """Build gap tables for multiple subgroup dimensions.

    Returns dict mapping group_col -> DataFrame of gaps.
    """
    tables = {}
    for col in group_cols:
        if col in df.columns:
            tables[col] = gap_by_subgroup(df, col, outcome, weight)
        else:
            logger.warning("Column %s not in DataFrame, skipping", col)
    return tables


def weighted_median_gap(
    df: pd.DataFrame,
    outcome: str = "hourly_wage_real",
    weight: str = "person_weight",
) -> dict:
    """Compute weighted median gap."""
    male = df[df["female"] == 0]
    female = df[df["female"] == 1]

    m_med = weighted_quantile(male[outcome], male[weight], 0.5)
    f_med = weighted_quantile(female[outcome], female[weight], 0.5)
    gap = m_med - f_med
    gap_pct = gap / m_med * 100 if m_med != 0 else float("nan")

    return {
        "male_median": m_med,
        "female_median": f_med,
        "gap_dollars": gap,
        "gap_pct": gap_pct,
    }


def raw_gap_with_sdr(
    df: pd.DataFrame,
    outcome: str = "hourly_wage_real",
    weight: str = "person_weight",
    repweight_prefix: str = "PWGTP",
) -> dict[str, Any]:
    """Compute ACS raw gap with SDR uncertainty from replicate weights."""
    point = raw_gap(df, outcome=outcome, weight=weight)
    rep_cols = replicate_weight_columns(df.columns, prefix=repweight_prefix)
    if not rep_cols:
        raise ValueError("No ACS replicate-weight columns found")

    male = df[df["female"] == 0]
    female = df[df["female"] == 1]
    rep_estimates = []
    for rep_col in rep_cols:
        m_mean = weighted_mean(male[outcome], male[rep_col])
        f_mean = weighted_mean(female[outcome], female[rep_col])
        gap_pct = ((m_mean - f_mean) / m_mean * 100) if m_mean != 0 else float("nan")
        rep_estimates.append(gap_pct)

    summary = sdr_summary(point["gap_pct"], rep_estimates)
    return {
        **point,
        "gap_pct_se": summary["se"],
        "gap_pct_moe90": summary["moe90"],
        "gap_pct_ci90_low": summary["ci90_low"],
        "gap_pct_ci90_high": summary["ci90_high"],
        "gap_pct_ci95_low": summary["ci95_low"],
        "gap_pct_ci95_high": summary["ci95_high"],
        "n_replicates": len(rep_cols),
    }
