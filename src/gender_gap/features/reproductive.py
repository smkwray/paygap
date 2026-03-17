"""Reproductive-burden feature engineering for ACS-style annual data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def add_reproductive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the seeded reproductive feature set exists on an annual table."""
    out = df.copy()

    noc = pd.to_numeric(
        _column_or_default(out, "noc", "number_children", "NOC"),
        errors="coerce",
    ).fillna(0)
    paoc = pd.to_numeric(_column_or_default(out, "paoc", "PAOC"), errors="coerce").fillna(0)
    fer = pd.to_numeric(_column_or_default(out, "fer", "FER"), errors="coerce").fillna(0)
    marhm = pd.to_numeric(_column_or_default(out, "marhm", "MARHM"), errors="coerce").fillna(0)
    cplt = pd.to_numeric(_column_or_default(out, "cplt", "CPLT"), errors="coerce").fillna(0)
    partner = pd.to_numeric(_column_or_default(out, "partner", "PARTNER"), errors="coerce").fillna(0)

    out["noc"] = noc
    out["paoc"] = paoc
    out["fer"] = fer
    out["marhm"] = marhm
    out["cplt"] = cplt
    out["partner"] = partner
    out["number_children"] = pd.to_numeric(
        out.get("number_children", noc), errors="coerce"
    ).fillna(noc)
    out["children_under_5"] = pd.to_numeric(
        out.get("children_under_5", paoc.isin([1, 3]).astype(int)),
        errors="coerce",
    ).fillna(0)

    out["recent_birth"] = _as_int(out.get("recent_birth"), default=(fer == 1))
    out["recent_marriage"] = _as_int(out.get("recent_marriage"), default=marhm.between(1, 12))
    out["has_own_child"] = _as_int(
        out.get("has_own_child"),
        default=(noc.gt(0) | paoc.isin([1, 2, 3])),
    )
    out["own_child_under6"] = _as_int(out.get("own_child_under6"), default=paoc.isin([1, 3]))
    out["own_child_6_17_only"] = _as_int(
        out.get("own_child_6_17_only"),
        default=paoc.eq(2),
    )
    out["same_sex_couple_household"] = _as_int(
        out.get("same_sex_couple_household"),
        default=cplt.isin([2, 4]),
    )
    out["opposite_sex_couple_household"] = _as_int(
        out.get("opposite_sex_couple_household"),
        default=cplt.isin([1, 3]),
    )
    out["couple_type"] = _couple_type(out)
    out["reproductive_stage"] = _reproductive_stage(out)
    out["age_fertility_band"] = _age_fertility_band(pd.to_numeric(out["age"], errors="coerce"))
    out["older_low_fertility_placebo"] = (
        pd.to_numeric(out["age"], errors="coerce").between(45, 54, inclusive="both")
    ).astype(int)
    out["parenthood_category"] = _legacy_parenthood_category(out)
    out["employment_indicator"] = out.get("employment_indicator", 1)
    out["ftfy_indicator"] = _ftfy_indicator(out)
    out["log_annual_earnings_real"] = _log_series(out.get("annual_earnings_real"))
    return out


def add_fertility_risk_features(
    df: pd.DataFrame,
    weight_col: str = "person_weight",
) -> pd.DataFrame:
    """Estimate a predicted recent-birth risk for childless women."""
    out = add_reproductive_features(df)
    out["fertility_risk_score"] = np.nan
    out["fertility_risk_quartile"] = pd.Series(pd.NA, index=out.index, dtype="string")

    age = pd.to_numeric(out["age"], errors="coerce")
    eligible = (
        out["female"].eq(1)
        & age.between(25, 44, inclusive="both")
        & out["has_own_child"].eq(0)
        & out["recent_birth"].notna()
    )
    if eligible.sum() < 4 or out.loc[eligible, "recent_birth"].nunique(dropna=True) < 2:
        logger.warning("Fertility-risk score skipped: insufficient eligible variation")
        return out

    predictor_names = [
        "age",
        "age_sq",
        "education_level",
        "race_ethnicity",
        "state_fips",
        "marital_status",
        "recent_marriage",
        "couple_type",
    ]
    train = out.loc[eligible, predictor_names + ["recent_birth", weight_col]].copy()
    train["age_sq"] = train.get("age_sq", train["age"] ** 2)
    features = pd.get_dummies(train[predictor_names], dummy_na=True, dtype=float)
    target = train["recent_birth"].astype(int)
    weights = pd.to_numeric(train[weight_col], errors="coerce").fillna(1.0)

    model = LogisticRegression(max_iter=1000)
    model.fit(features, target, sample_weight=weights)

    childless = out["female"].eq(1) & out["has_own_child"].eq(0)
    score_frame = out.loc[childless, predictor_names].copy()
    score_frame["age_sq"] = score_frame.get("age_sq", score_frame["age"] ** 2)
    score_X = pd.get_dummies(score_frame, dummy_na=True, dtype=float).reindex(
        columns=features.columns,
        fill_value=0.0,
    )
    scores = pd.Series(
        model.predict_proba(score_X)[:, 1],
        index=score_frame.index,
        dtype=float,
    )
    out.loc[scores.index, "fertility_risk_score"] = scores

    eligible_scores = scores.loc[eligible.loc[scores.index]]
    if eligible_scores.notna().sum() >= 4:
        ranked = eligible_scores.rank(method="first")
        quartiles = pd.qcut(ranked, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        out.loc[quartiles.index, "fertility_risk_quartile"] = quartiles.astype("string")
        older = childless & age.between(45, 54, inclusive="both")
        if older.any():
            older_scores = scores.loc[older.loc[scores.index]]
            older_ranked = older_scores.rank(method="first")
            older_quartiles = pd.qcut(
                older_ranked,
                min(4, older_ranked.notna().sum()),
                labels=["Q1", "Q2", "Q3", "Q4"][: min(4, older_ranked.notna().sum())],
            )
            out.loc[older_quartiles.index, "fertility_risk_quartile"] = older_quartiles.astype(
                "string"
            )
    return out


def add_repro_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Build the slim interaction set used by the seeded M8 surface."""
    out = df.copy()
    female_source = out["female"] if "female" in out.columns else pd.Series(0, index=out.index)
    female = pd.to_numeric(female_source, errors="coerce").fillna(0)
    terms = [
        "recent_birth",
        "own_child_under6",
        "recent_marriage",
        "same_sex_couple_household",
        "autonomy",
        "job_rigidity",
    ]
    for term in terms:
        if term in out.columns:
            out[f"female_x_{term}"] = female * pd.to_numeric(out[term], errors="coerce").fillna(0)
    if {"own_child_under6", "job_rigidity"}.issubset(out.columns):
        own_child = pd.to_numeric(out["own_child_under6"], errors="coerce").fillna(0)
        rigidity = pd.to_numeric(out["job_rigidity"], errors="coerce").fillna(0)
        out["female_x_own_child_under6_x_job_rigidity"] = female * own_child * rigidity
    return out


def _as_int(values, default) -> pd.Series:
    if values is None:
        return pd.Series(default, dtype="Int64").astype(int)
    series = pd.to_numeric(values, errors="coerce")
    if series.notna().any():
        return series.fillna(pd.Series(default, index=series.index).astype(float)).astype(int)
    return pd.Series(default, index=getattr(values, "index", None)).astype(int)


def _column_or_default(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return df[name]
    return pd.Series(0, index=df.index, dtype=float)


def _couple_type(df: pd.DataFrame) -> pd.Series:
    result = pd.Series("unpartnered", index=df.index, dtype="string")
    result.loc[df["opposite_sex_couple_household"] == 1] = "opposite_sex"
    result.loc[df["same_sex_couple_household"] == 1] = "same_sex"
    return result


def _reproductive_stage(df: pd.DataFrame) -> pd.Series:
    result = pd.Series("childless_unpartnered", index=df.index, dtype="string")
    childless = df["has_own_child"].eq(0)
    partnered = df["couple_type"].ne("unpartnered")
    result.loc[childless & partnered] = "childless_other_partnered"
    result.loc[childless & df["recent_marriage"].eq(1)] = "childless_recently_married"
    result.loc[df["own_child_6_17_only"].eq(1)] = "mother_6_17_only"
    result.loc[df["own_child_under6"].eq(1)] = "mother_under6"
    result.loc[
        df["has_own_child"].eq(1)
        & ~df["own_child_under6"].eq(1)
        & ~df["own_child_6_17_only"].eq(1)
    ] = "mother_mixed_or_other"
    result.loc[df["recent_birth"].eq(1)] = "recent_birth"
    return result


def _age_fertility_band(age: pd.Series) -> pd.Series:
    result = pd.Series("outside_range", index=age.index, dtype="string")
    for lo, hi, label in [
        (25, 29, "25_29"),
        (30, 34, "30_34"),
        (35, 39, "35_39"),
        (40, 44, "40_44"),
        (45, 49, "45_49"),
        (50, 54, "50_54"),
    ]:
        result.loc[age.between(lo, hi, inclusive="both")] = label
    return result


def _legacy_parenthood_category(df: pd.DataFrame) -> pd.Series:
    result = pd.Series("no_children", index=df.index, dtype="string")
    result.loc[df["has_own_child"].eq(1)] = "has_children"
    result.loc[df["own_child_under6"].eq(1)] = "young_children"
    return result


def _ftfy_indicator(df: pd.DataFrame) -> pd.Series:
    weeks = pd.to_numeric(_column_or_default(df, "weeks_worked"), errors="coerce").fillna(0)
    hours = pd.to_numeric(_column_or_default(df, "usual_hours_week"), errors="coerce").fillna(0)
    return ((weeks >= 50) & (hours >= 35)).astype(int)


def _log_series(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    series = pd.to_numeric(values, errors="coerce")
    if not isinstance(series, pd.Series):
        if pd.isna(series) or series <= 0:
            return pd.Series(dtype=float)
        return pd.Series([np.log(series)], dtype=float)
    if series is None:
        return pd.Series(dtype=float)
    series = series.copy()
    series[series <= 0] = np.nan
    return np.log(series)
