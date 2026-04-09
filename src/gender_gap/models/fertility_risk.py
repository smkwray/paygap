"""Women-trained recent-birth proxy models for reproductive analyses."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gender_gap.models.ols import coefficient_table
from gender_gap.utils.weights import weighted_mean

logger = logging.getLogger(__name__)


def run_fertility_risk_penalty(
    df: pd.DataFrame,
    weight_col: str = "person_weight",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate women-trained proxy outputs for main and negative-control samples."""
    women_main = _childless_sample(df, 25, 44, female=1)
    if women_main.empty or women_main["fertility_risk_quartile"].dropna().empty:
        empty = pd.DataFrame(columns=["sample", "outcome", "term", "coef", "se", "pvalue"])
        quartiles = pd.DataFrame(
            columns=["sample", "outcome", "risk_quartile", "mean_outcome", "n_obs", "weighted_n"]
        )
        return empty, quartiles

    results = []
    quartile_frames = []
    for label, sample in [
        ("childless_25_44", women_main),
        ("childless_men_25_44", _childless_sample(df, 25, 44, female=0)),
    ]:
        if sample.empty or sample["fertility_risk_quartile"].dropna().empty:
            continue
        penalty_table = _penalty_table(sample, label, weight_col)
        if not penalty_table.empty:
            results.append(penalty_table)
        quartile_table = _quartile_means(sample, weight_col)
        if not quartile_table.empty:
            quartile_table.insert(0, "sample", label)
            quartile_frames.append(quartile_table)

    for lo, hi, label in [(45, 49, "childless_45_49"), (50, 54, "childless_50_54")]:
        placebo = _childless_sample(df, lo, hi, female=1)
        if placebo.empty:
            continue
        placebo_table = _quartile_means(placebo, weight_col)
        placebo_table.insert(0, "sample", label)
        quartile_frames.append(placebo_table)

    result_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    quartile_df = pd.concat(quartile_frames, ignore_index=True) if quartile_frames else pd.DataFrame()
    return result_df, quartile_df


def build_same_sex_placebos(df: pd.DataFrame, weight_col: str = "person_weight") -> pd.DataFrame:
    """Create a compact placebo/contrast table for same-sex and older-age groups."""
    rows = []
    samples = {
        "main_childless_25_44": _childless_sample(df, 25, 44, female=1),
        "placebo_childless_45_49": _childless_sample(df, 45, 49, female=1),
        "placebo_childless_50_54": _childless_sample(df, 50, 54, female=1),
    }
    for label, sample in samples.items():
        if sample.empty or "same_sex_couple_household" not in sample.columns:
            continue
        for status, sdf in sample.groupby("same_sex_couple_household", dropna=False):
            rows.append(
                {
                    "sample": label,
                    "same_sex_couple_household": int(status),
                    "n_obs": int(len(sdf)),
                    "mean_hourly_wage": _weighted_mean(sdf["hourly_wage_real"], sdf[weight_col]),
                    "mean_hours": _weighted_mean(sdf["usual_hours_week"], sdf[weight_col]),
                    "mean_recent_birth": _weighted_mean(sdf["recent_birth"], sdf[weight_col]),
                }
            )
    return pd.DataFrame(rows)


def _penalty_table(df: pd.DataFrame, sample_label: str, weight_col: str) -> pd.DataFrame:
    outcomes = [
        "log_hourly_wage_real",
        "log_annual_earnings_real",
        "usual_hours_week",
        "employment_indicator",
        "ftfy_indicator",
    ]
    tables = []
    for outcome in outcomes:
        if outcome == "log_hourly_wage_real" and outcome not in df.columns and "hourly_wage_real" in df.columns:
            df = df.copy()
            hourly = pd.to_numeric(df["hourly_wage_real"], errors="coerce")
            df["log_hourly_wage_real"] = np.log(hourly.where(hourly > 0))
        if outcome not in df.columns or df[outcome].notna().sum() < 20:
            continue
        design = df.copy()
        quartiles = pd.get_dummies(
            design["fertility_risk_quartile"],
            prefix="fertility_risk_quartile",
            drop_first=True,
            dtype=float,
        )
        if quartiles.empty:
            continue
        design = pd.concat([design, quartiles], axis=1)
        controls = [
            "age",
            "age_sq",
            "recent_marriage",
            "C(education_level)",
            "C(race_ethnicity)",
            "C(state_fips)",
            "C(marital_status)",
        ] + list(quartiles.columns)
        table = coefficient_table(
            design,
            model_name="fertility_risk",
            outcome=outcome,
            weight_col=weight_col,
            blocks={"fertility_risk": controls},
        )
        table.insert(0, "sample", sample_label)
        table.insert(1, "outcome", outcome)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _childless_sample(
    df: pd.DataFrame,
    age_min: int,
    age_max: int,
    female: int | None = None,
) -> pd.DataFrame:
    age = pd.to_numeric(df.get("age"), errors="coerce")
    mask = (
        df.get("has_own_child", pd.Series(0, index=df.index)).eq(0)
        & age.between(age_min, age_max, inclusive="both")
    )
    if female is not None:
        mask &= df.get("female", pd.Series(0, index=df.index)).eq(female)
    return df.loc[mask].copy()


def _quartile_means(df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    rows = []
    for outcome in ["hourly_wage_real", "annual_earnings_real", "usual_hours_week"]:
        if outcome not in df.columns:
            continue
        for quartile, qdf in df.groupby("fertility_risk_quartile", observed=True):
            if pd.isna(quartile):
                continue
            values = pd.to_numeric(qdf[outcome], errors="coerce")
            weights = pd.to_numeric(qdf[weight_col], errors="coerce")
            mask = values.notna() & weights.notna() & (weights > 0)
            if not mask.any():
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "risk_quartile": quartile,
                    "mean_outcome": weighted_mean(values[mask], weights[mask]),
                    "n_obs": int(mask.sum()),
                    "weighted_n": float(weights[mask].sum()),
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _weighted_mean(values, weights) -> float:
    series = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = series.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float((series[mask] * w[mask]).sum() / w[mask].sum())
