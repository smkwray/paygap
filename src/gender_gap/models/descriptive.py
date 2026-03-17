"""Descriptive weighted gap tables.

Computes weighted mean and median wage gaps by subgroups.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from gender_gap.models.ols import coefficient_table
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


def build_lesbian_married_summary(
    df: pd.DataFrame,
    weight: str = "person_weight",
) -> pd.DataFrame:
    """Summarize married lesbian couples and key comparison groups.

    The output is a long-form table with:
    - `section=summary` rows for weighted subgroup means
    - `section=comparison` rows for weighted differences vs comparison groups
    """
    groups = {
        "lesbian_married": (
            df["female"].eq(1)
            & df["same_sex_couple_household"].eq(1)
            & df["marital_status"].eq("married")
        ),
        "women_opposite_sex_married": (
            df["female"].eq(1)
            & df["same_sex_couple_household"].eq(0)
            & df["marital_status"].eq("married")
        ),
        "gay_married": (
            df["female"].eq(0)
            & df["same_sex_couple_household"].eq(1)
            & df["marital_status"].eq("married")
        ),
        "men_opposite_sex_married": (
            df["female"].eq(0)
            & df["same_sex_couple_household"].eq(0)
            & df["marital_status"].eq("married")
        ),
    }

    group_stats: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    metric_specs = {
        "mean_hourly_wage": "hourly_wage_real",
        "mean_annual_earnings": "annual_earnings_real",
        "mean_usual_hours_week": "usual_hours_week",
        "share_recent_birth": "recent_birth",
        "share_recent_marriage": "recent_marriage",
        "share_own_child_under6": "own_child_under6",
    }

    for name, mask in groups.items():
        gdf = df.loc[mask].copy()
        weighted_n = float(pd.to_numeric(gdf.get(weight), errors="coerce").fillna(0).sum())
        stats = {
            "n_obs": int(len(gdf)),
            "weighted_n": weighted_n,
        }
        rows.append(
            {
                "section": "summary",
                "group": name,
                "comparison_group": "",
                "metric": "n_obs",
                "value": int(len(gdf)),
                "n_obs": int(len(gdf)),
                "weighted_n": weighted_n,
            }
        )
        rows.append(
            {
                "section": "summary",
                "group": name,
                "comparison_group": "",
                "metric": "weighted_n",
                "value": weighted_n,
                "n_obs": int(len(gdf)),
                "weighted_n": weighted_n,
            }
        )
        for metric_name, column in metric_specs.items():
            sample = gdf.loc[pd.to_numeric(gdf.get(column), errors="coerce").notna()].copy()
            stats[metric_name] = weighted_mean(sample[column], sample[weight]) if not sample.empty else float("nan")
            rows.append(
                {
                    "section": "summary",
                    "group": name,
                    "comparison_group": "",
                    "metric": metric_name,
                    "value": stats[metric_name],
                    "n_obs": int(len(gdf)),
                    "weighted_n": weighted_n,
                }
            )
        group_stats[name] = stats

    comparisons = [
        ("lesbian_married", "women_opposite_sex_married"),
        ("lesbian_married", "gay_married"),
        ("lesbian_married", "men_opposite_sex_married"),
    ]
    comparison_metrics = {
        "hourly_wage_gap_dollars": ("mean_hourly_wage", False),
        "hourly_wage_gap_pct": ("mean_hourly_wage", True),
        "annual_earnings_gap_dollars": ("mean_annual_earnings", False),
        "annual_earnings_gap_pct": ("mean_annual_earnings", True),
        "hours_gap": ("mean_usual_hours_week", False),
    }
    for focal, comp in comparisons:
        focal_stats = group_stats.get(focal, {})
        comp_stats = group_stats.get(comp, {})
        for metric, (base_metric, pct) in comparison_metrics.items():
            focal_value = float(focal_stats.get(base_metric, float("nan")))
            comp_value = float(comp_stats.get(base_metric, float("nan")))
            if pct:
                value = (
                    (focal_value - comp_value) / comp_value * 100.0
                    if pd.notna(comp_value) and comp_value != 0
                    else float("nan")
                )
            else:
                value = focal_value - comp_value
            rows.append(
                {
                    "section": "comparison",
                    "group": focal,
                    "comparison_group": comp,
                    "metric": metric,
                    "value": value,
                    "n_obs": int(focal_stats.get("n_obs", 0)),
                    "weighted_n": float(focal_stats.get("weighted_n", float("nan"))),
                }
            )

    return pd.DataFrame(rows)


def build_lesbian_married_adjusted_table(
    df: pd.DataFrame,
    weight_col: str = "person_weight",
) -> pd.DataFrame:
    """Estimate adjusted lesbian-married coefficients among married women.

    This is a negative-control style comparison:
    - sample restricted to married women
    - coefficient is lesbian-married vs married non-same-sex women
    """
    sample = df.loc[
        df["female"].eq(1)
        & df["marital_status"].eq("married")
        & df["same_sex_couple_household"].isin([0, 1])
    ].copy()
    if sample.empty:
        return pd.DataFrame(
            columns=[
                "sample",
                "outcome",
                "model",
                "term",
                "coef",
                "se",
                "pvalue",
                "n_obs",
                "r_squared",
                "pct_effect",
            ]
        )

    sample["lesbian_married"] = sample["same_sex_couple_household"].fillna(0).astype(int)
    if "log_hourly_wage_real" not in sample.columns:
        sample["log_hourly_wage_real"] = np.log(
            pd.to_numeric(sample["hourly_wage_real"], errors="coerce").where(
                pd.to_numeric(sample["hourly_wage_real"], errors="coerce") > 0
            )
        )
    if "log_annual_earnings_real" not in sample.columns:
        sample["log_annual_earnings_real"] = np.log(
            pd.to_numeric(sample["annual_earnings_real"], errors="coerce").where(
                pd.to_numeric(sample["annual_earnings_real"], errors="coerce") > 0
            )
        )
    sample["age_sq"] = pd.to_numeric(sample["age"], errors="coerce") ** 2

    blocks = {
        "L0_raw": ["lesbian_married"],
        "L1_demographics": ["lesbian_married", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
        "L2_job_sorting": [
            "lesbian_married", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)",
        ],
        "L3_work_arrangement": [
            "lesbian_married", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)",
            "usual_hours_week", "work_from_home", "commute_minutes_one_way",
        ],
        "L4_reproductive": [
            "lesbian_married", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)",
            "usual_hours_week", "work_from_home", "commute_minutes_one_way",
            "number_children", "children_under_5", "recent_birth", "recent_marriage",
            "has_own_child", "own_child_under6", "own_child_6_17_only", "C(reproductive_stage)",
        ],
        "L5_onet_context": [
            "lesbian_married", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
            "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)",
            "usual_hours_week", "work_from_home", "commute_minutes_one_way",
            "number_children", "children_under_5", "recent_birth", "recent_marriage",
            "has_own_child", "own_child_under6", "own_child_6_17_only", "C(reproductive_stage)",
            "autonomy", "schedule_unpredictability", "time_pressure",
            "coordination_responsibility", "physical_proximity", "job_rigidity",
        ],
    }

    outputs: list[pd.DataFrame] = []
    for outcome in ["log_hourly_wage_real", "log_annual_earnings_real", "usual_hours_week"]:
        for model_name in blocks:
            try:
                fitted = coefficient_table(
                    sample,
                    model_name=model_name,
                    outcome=outcome,
                    weight_col=weight_col,
                    blocks=blocks,
                )
            except Exception:
                continue
            term_df = fitted.loc[fitted["term"] == "lesbian_married"].copy()
            if term_df.empty:
                continue
            term_df.insert(0, "sample", "married_women_negative_control")
            term_df.insert(1, "outcome", outcome)
            term_df["pct_effect"] = np.where(
                term_df["outcome"].str.startswith("log_"),
                (np.exp(term_df["coef"]) - 1.0) * 100.0,
                np.nan,
            )
            outputs.append(term_df)

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


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
