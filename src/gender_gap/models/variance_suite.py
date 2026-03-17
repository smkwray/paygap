"""Variance and tail diagnostics for distribution-focused gap analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

VARIANCE_METRICS = [
    "raw_variance_ratio",
    "residual_variance_ratio",
    "male_p90_p10",
    "female_p90_p10",
    "male_p95_p50",
    "female_p95_p50",
    "male_top10_share",
    "female_top10_share",
    "male_top5_share",
    "female_top5_share",
]

DEFAULT_STRATIFIERS = [
    "overall",
    "reproductive_stage",
    "fertility_risk_quartile",
    "job_rigidity_quartile",
    "couple_type",
]


def run_variance_suite(
    df: pd.DataFrame,
    outcome: str = "log_hourly_wage_real",
    weight_col: str = "person_weight",
    stratifiers: list[str] | None = None,
    min_group_n: int | None = None,
    max_groups: int | None = None,
) -> pd.DataFrame:
    """Build variance and upper-tail diagnostics by sex across stratifiers.

    Notes
    -----
    - `residual_variance_ratio` uses a fast weighted within-cell residualization
      (survey year x age-band x race x education) to avoid dense dummy models on
      multi-million-row ACS panels.
    - top-tail shares use pooled thresholds within each stratum and then compute
      sex-specific weighted shares above those pooled cutoffs.
    """
    if stratifiers is None:
        stratifiers = DEFAULT_STRATIFIERS

    frame = _prepare_analysis_frame(df, outcome=outcome, weight_col=weight_col)
    if frame.empty:
        return pd.DataFrame(columns=["stratifier", "stratum", "metric", "value", "n_obs"])

    outcome_col = "__outcome"
    frame["__residual"] = _residualize_outcome(frame, outcome=outcome_col, weight_col=weight_col)

    records: list[dict] = []
    for stratifier in stratifiers:
        groups = _iter_groups(
            frame,
            stratifier=stratifier,
            weight_col=weight_col,
            max_groups=max_groups,
        )
        for stratum, gdf in groups:
            n_obs = int(len(gdf))
            if stratifier != "overall" and min_group_n is not None and n_obs < int(min_group_n):
                continue
            metrics = _group_metrics(gdf, outcome_col=outcome_col, weight=weight_col)
            for metric in VARIANCE_METRICS:
                records.append(
                    {
                        "stratifier": stratifier,
                        "stratum": str(stratum),
                        "metric": metric,
                        "value": float(metrics.get(metric, np.nan)),
                        "n_obs": n_obs,
                    }
                )
    return pd.DataFrame.from_records(records)


def _prepare_analysis_frame(df: pd.DataFrame, outcome: str, weight_col: str) -> pd.DataFrame:
    frame = df.copy()
    if (
        outcome not in frame.columns
        and outcome == "log_hourly_wage_real"
        and "hourly_wage_real" in frame.columns
    ):
        hourly = pd.to_numeric(frame["hourly_wage_real"], errors="coerce")
        frame["log_hourly_wage_real"] = np.log(hourly.where(hourly > 0))
    if (
        outcome not in frame.columns
        and outcome == "log_annual_earnings_real"
        and "annual_earnings_real" in frame.columns
    ):
        annual = pd.to_numeric(frame["annual_earnings_real"], errors="coerce")
        frame["log_annual_earnings_real"] = np.log(annual.where(annual > 0))

    if (
        outcome not in frame.columns
        or weight_col not in frame.columns
        or "female" not in frame.columns
    ):
        return pd.DataFrame()

    frame["__outcome"] = pd.to_numeric(frame[outcome], errors="coerce")
    frame[weight_col] = pd.to_numeric(frame[weight_col], errors="coerce")
    frame["female"] = pd.to_numeric(frame["female"], errors="coerce")
    mask = (
        frame["__outcome"].notna()
        & frame[weight_col].notna()
        & frame[weight_col].gt(0)
        & frame["female"].isin([0, 1])
    )
    return frame.loc[mask].copy()


def _residualize_outcome(df: pd.DataFrame, outcome: str, weight_col: str) -> pd.Series:
    temp = pd.DataFrame(index=df.index)
    temp[outcome] = pd.to_numeric(df[outcome], errors="coerce")
    temp[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")

    group_cols: list[str] = []
    if "survey_year" in df.columns:
        temp["survey_year"] = df["survey_year"]
        group_cols.append("survey_year")
    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        bins = [-np.inf, 24, 29, 34, 39, 44, 49, 54, 59, np.inf]
        labels = ["u24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60p"]
        temp["age_band"] = pd.cut(age, bins=bins, labels=labels, right=True).astype("string")
        group_cols.append("age_band")
    for column in ["race_ethnicity", "education_level"]:
        if column in df.columns:
            temp[column] = df[column].astype("string").fillna("missing")
            group_cols.append(column)

    if not group_cols:
        centered = temp[outcome] - _weighted_mean(temp[outcome], temp[weight_col])
        return centered.astype(float)

    temp["__wy"] = temp[outcome] * temp[weight_col]
    grouped = (
        temp.groupby(group_cols, dropna=False, observed=True)[[weight_col, "__wy"]]
        .sum()
        .rename(columns={weight_col: "__w_sum", "__wy": "__wy_sum"})
        .reset_index()
    )
    merged = temp[group_cols + [outcome]].merge(grouped, on=group_cols, how="left")
    fitted = merged["__wy_sum"] / merged["__w_sum"]
    residual = merged[outcome] - fitted
    residual.index = temp.index
    return residual.astype(float)


def _iter_groups(
    df: pd.DataFrame,
    stratifier: str,
    weight_col: str,
    max_groups: int | None,
):
    if stratifier == "overall":
        return [("all", df)]
    if stratifier not in df.columns:
        return []

    grouped = (
        df.assign(__stratum=df[stratifier].astype("string").fillna("missing"))
        .groupby("__stratum", dropna=False, observed=True, sort=False)
    )
    items = [(stratum, gdf.copy()) for stratum, gdf in grouped]
    if max_groups is None or len(items) <= max_groups:
        return items

    ranked = sorted(
        items,
        key=lambda pair: float(pd.to_numeric(pair[1][weight_col], errors="coerce").sum()),
        reverse=True,
    )
    return ranked[: int(max_groups)]


def _group_metrics(gdf: pd.DataFrame, outcome_col: str, weight: str) -> dict[str, float]:
    female_mask = pd.to_numeric(gdf["female"], errors="coerce").eq(1)
    male_mask = pd.to_numeric(gdf["female"], errors="coerce").eq(0)

    y = pd.to_numeric(gdf[outcome_col], errors="coerce")
    w = pd.to_numeric(gdf[weight], errors="coerce")
    resid = pd.to_numeric(gdf["__residual"], errors="coerce")

    y_f, w_f = y.loc[female_mask], w.loc[female_mask]
    y_m, w_m = y.loc[male_mask], w.loc[male_mask]
    r_f, rw_f = resid.loc[female_mask], w.loc[female_mask]
    r_m, rw_m = resid.loc[male_mask], w.loc[male_mask]

    raw_var_f = _weighted_variance(y_f, w_f)
    raw_var_m = _weighted_variance(y_m, w_m)
    resid_var_f = _weighted_variance(r_f, rw_f)
    resid_var_m = _weighted_variance(r_m, rw_m)

    pooled_p90 = _weighted_quantile(y, w, 0.90)
    pooled_p95 = _weighted_quantile(y, w, 0.95)

    male_p95 = _weighted_quantile(y_m, w_m, 0.95)
    male_p90 = _weighted_quantile(y_m, w_m, 0.90)
    male_p50 = _weighted_quantile(y_m, w_m, 0.50)
    male_p10 = _weighted_quantile(y_m, w_m, 0.10)
    female_p95 = _weighted_quantile(y_f, w_f, 0.95)
    female_p90 = _weighted_quantile(y_f, w_f, 0.90)
    female_p50 = _weighted_quantile(y_f, w_f, 0.50)
    female_p10 = _weighted_quantile(y_f, w_f, 0.10)

    return {
        "raw_variance_ratio": _safe_ratio(raw_var_f, raw_var_m),
        "residual_variance_ratio": _safe_ratio(resid_var_f, resid_var_m),
        "male_p90_p10": (
            male_p90 - male_p10 if pd.notna(male_p90) and pd.notna(male_p10) else np.nan
        ),
        "female_p90_p10": (
            female_p90 - female_p10 if pd.notna(female_p90) and pd.notna(female_p10) else np.nan
        ),
        "male_p95_p50": (
            male_p95 - male_p50 if pd.notna(male_p95) and pd.notna(male_p50) else np.nan
        ),
        "female_p95_p50": (
            female_p95 - female_p50 if pd.notna(female_p95) and pd.notna(female_p50) else np.nan
        ),
        "male_top10_share": _top_share(y_m, w_m, pooled_p90),
        "female_top10_share": _top_share(y_f, w_f, pooled_p90),
        "male_top5_share": _top_share(y_m, w_m, pooled_p95),
        "female_top5_share": _top_share(y_f, w_f, pooled_p95),
    }


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & w.gt(0)
    if not mask.any():
        return float("nan")
    vv = v.loc[mask].to_numpy(dtype=float, copy=False)
    ww = w.loc[mask].to_numpy(dtype=float, copy=False)
    return float(np.average(vv, weights=ww))


def _weighted_variance(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & w.gt(0)
    if mask.sum() < 2:
        return float("nan")
    vv = v.loc[mask].to_numpy(dtype=float, copy=False)
    ww = w.loc[mask].to_numpy(dtype=float, copy=False)
    mu = np.average(vv, weights=ww)
    return float(np.average((vv - mu) ** 2, weights=ww))


def _weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & w.gt(0)
    if not mask.any():
        return float("nan")
    vv = v.loc[mask].to_numpy(dtype=float, copy=False)
    ww = w.loc[mask].to_numpy(dtype=float, copy=False)
    order = np.argsort(vv)
    vv = vv[order]
    ww = ww[order]
    cdf = np.cumsum(ww)
    if cdf[-1] <= 0:
        return float("nan")
    threshold = q * cdf[-1]
    idx = np.searchsorted(cdf, threshold, side="left")
    idx = min(max(int(idx), 0), len(vv) - 1)
    return float(vv[idx])


def _top_share(values: pd.Series, weights: pd.Series, threshold: float) -> float:
    if pd.isna(threshold):
        return float("nan")
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & w.gt(0)
    if not mask.any():
        return float("nan")
    vv = v.loc[mask]
    ww = w.loc[mask]
    denom = float(ww.sum())
    if denom <= 0:
        return float("nan")
    numer = float(ww.loc[vv >= threshold].sum())
    return numer / denom


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)
