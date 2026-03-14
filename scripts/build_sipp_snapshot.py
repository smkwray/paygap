#!/usr/bin/env python3
"""Build a first descriptive analysis surface for the standardized SIPP file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gender_gap.models.descriptive import raw_gap, weighted_median_gap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sipp_standardized.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
SIPP_RESULTS_DIR = RESULTS_DIR / "sipp" / "2023"
TRENDS_DIR = RESULTS_DIR / "trends"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _labor_force_share(df: pd.DataFrame, status: str) -> float:
    return float((df["labor_force_status"] == status).mean())


def build_snapshot_summary(df: pd.DataFrame) -> pd.DataFrame:
    hourly_workers = df[
        df["employed"].fillna(0).eq(1)
        & df["hourly_wage_real"].gt(0)
        & df["hourly_wage_real"].notna()
    ].copy()
    weekly_workers = df[
        df["employed"].fillna(0).eq(1)
        & df["weekly_earnings_real"].gt(0)
        & df["weekly_earnings_real"].notna()
    ].copy()

    hourly_raw = raw_gap(hourly_workers, outcome="hourly_wage_real", weight="person_weight")
    hourly_med = weighted_median_gap(hourly_workers, outcome="hourly_wage_real", weight="person_weight")
    weekly_raw = raw_gap(weekly_workers, outcome="weekly_earnings_real", weight="person_weight")
    weekly_med = weighted_median_gap(weekly_workers, outcome="weekly_earnings_real", weight="person_weight")

    rows = [
        ("rows", int(len(df))),
        ("months_covered", int(df["month"].nunique())),
        ("female_share", float(df["female"].mean())),
        ("employed_share", float(df["employed"].mean())),
        ("unemployed_share", _labor_force_share(df, "unemployed")),
        ("not_in_labor_force_share", _labor_force_share(df, "not_in_labor_force")),
        ("missing_labor_force_share", float(df["labor_force_status"].isna().mean())),
        ("hourly_worker_rows", int(len(hourly_workers))),
        ("hourly_raw_gap_pct", float(hourly_raw["gap_pct"])),
        ("hourly_raw_gap_dollars", float(hourly_raw["gap_dollars"])),
        ("hourly_median_gap_pct", float(hourly_med["gap_pct"])),
        ("weekly_worker_rows", int(len(weekly_workers))),
        ("weekly_raw_gap_pct", float(weekly_raw["gap_pct"])),
        ("weekly_raw_gap_dollars", float(weekly_raw["gap_dollars"])),
        ("weekly_median_gap_pct", float(weekly_med["gap_pct"])),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for month, mdf in df.groupby("month", observed=True):
        hourly_workers = mdf[
            mdf["employed"].fillna(0).eq(1)
            & mdf["hourly_wage_real"].gt(0)
            & mdf["hourly_wage_real"].notna()
        ].copy()
        weekly_workers = mdf[
            mdf["employed"].fillna(0).eq(1)
            & mdf["weekly_earnings_real"].gt(0)
            & mdf["weekly_earnings_real"].notna()
        ].copy()
        hourly_gap = raw_gap(hourly_workers, outcome="hourly_wage_real", weight="person_weight")
        weekly_gap = raw_gap(weekly_workers, outcome="weekly_earnings_real", weight="person_weight")
        rows.append(
            {
                "calendar_year": int(mdf["calendar_year"].dropna().iloc[0]),
                "month": int(month),
                "employed_share": float(mdf["employed"].mean()),
                "hourly_worker_n": int(len(hourly_workers)),
                "hourly_gap_pct": float(hourly_gap["gap_pct"]),
                "weekly_worker_n": int(len(weekly_workers)),
                "weekly_gap_pct": float(weekly_gap["gap_pct"]),
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def write_report(summary: pd.DataFrame, trend: pd.DataFrame, path: Path) -> Path:
    metric = summary.set_index("metric")["value"]
    peak_hourly = trend.loc[trend["hourly_gap_pct"].idxmax()]
    low_hourly = trend.loc[trend["hourly_gap_pct"].idxmin()]
    lines = [
        "# SIPP Snapshot",
        "",
        "This note gives the project a first descriptive SIPP analysis layer from the validated 2023 public-use file.",
        "It is intentionally a weighted snapshot, not a full SIPP model stack.",
        "",
        "## Headline snapshot",
        "",
        f"- Rows: {int(metric['rows']):,}",
        f"- Months covered: {int(metric['months_covered'])}",
        f"- Female share: {metric['female_share'] * 100:.2f}%",
        f"- Employed share: {metric['employed_share'] * 100:.2f}%",
        f"- Unemployed share: {metric['unemployed_share'] * 100:.2f}%",
        f"- Not in labor force share: {metric['not_in_labor_force_share'] * 100:.2f}%",
        "",
        "## Worker gap snapshot",
        "",
        f"- Hourly wage raw gap: {metric['hourly_raw_gap_pct']:.2f}% ({metric['hourly_raw_gap_dollars']:.2f} dollars)",
        f"- Hourly wage median gap: {metric['hourly_median_gap_pct']:.2f}%",
        f"- Weekly earnings raw gap: {metric['weekly_raw_gap_pct']:.2f}% ({metric['weekly_raw_gap_dollars']:.2f} dollars)",
        f"- Weekly earnings median gap: {metric['weekly_median_gap_pct']:.2f}%",
        "",
        "## Monthly pattern",
        "",
        f"- Highest hourly-gap month: {int(peak_hourly['month'])} ({peak_hourly['hourly_gap_pct']:.2f}%)",
        f"- Lowest hourly-gap month: {int(low_hourly['month'])} ({low_hourly['hourly_gap_pct']:.2f}%)",
        "",
        "## Interpretation",
        "",
        "- SIPP now contributes more than a raw staged file: it has a canonical descriptive artifact for the latest public-use release.",
        "- The estimates here should be treated as descriptive monthly worker-gap evidence, not as directly interchangeable with the ACS annual wage regressions.",
        "- If the project extends SIPP further, the next honest step is a modest SIPP-specific model module built off this validated 2023 surface.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)
    summary = build_snapshot_summary(df)
    trend = build_monthly_trend(df)

    SIPP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRENDS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(SIPP_RESULTS_DIR / "snapshot_summary.csv", index=False)
    trend.to_csv(TRENDS_DIR / "sipp_monthly_gap_2023.csv", index=False)
    write_report(summary, trend, REPORTS_DIR / "sipp_snapshot.md")

    print(f"Wrote {SIPP_RESULTS_DIR / 'snapshot_summary.csv'}")
    print(f"Wrote {TRENDS_DIR / 'sipp_monthly_gap_2023.csv'}")
    print(f"Wrote {REPORTS_DIR / 'sipp_snapshot.md'}")


if __name__ == "__main__":
    main()
