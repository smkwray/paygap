"""Build validation artifacts for the standardized public-use SIPP file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "sipp_standardized.parquet"
DEFAULT_REPORT = ROOT / "reports" / "sipp_validation.md"
DEFAULT_SUMMARY = ROOT / "results" / "diagnostics" / "sipp_validation_summary.csv"


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    valid = series.notna() & weights.notna()
    if not valid.any():
        return float("nan")
    return float((series[valid] * weights[valid]).sum() / weights[valid].sum())


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    employed = df["employed"].fillna(0).eq(1)
    weights = pd.to_numeric(df["person_weight"], errors="coerce")

    metrics: list[tuple[str, float | int | str]] = [
        ("rows", int(len(df))),
        ("calendar_year_min", int(df["calendar_year"].min())),
        ("calendar_year_max", int(df["calendar_year"].max())),
        ("female_share", float(df["female"].mean())),
        ("employed_share", float(df["employed"].mean())),
        ("unemployed_share", float(df["labor_force_status"].eq("unemployed").mean())),
        ("not_in_labor_force_share", float(df["labor_force_status"].eq("not_in_labor_force").mean())),
        ("missing_labor_force_share", float(df["labor_force_status"].isna().mean())),
        ("weekly_earnings_nonnull_share", float(df["weekly_earnings_real"].notna().mean())),
        ("hourly_wage_nonnull_share", float(df["hourly_wage_real"].notna().mean())),
        ("usual_hours_nonnull_share", float(df["usual_hours_week"].notna().mean())),
        ("actual_hours_nonnull_share", float(df["actual_hours_last_week"].notna().mean())),
        ("occupation_nonnull_share_employed", float(df.loc[employed, "occupation_code"].notna().mean())),
        ("industry_nonnull_share_employed", float(df.loc[employed, "industry_code"].notna().mean())),
        ("paid_hourly_share_employed", float(df.loc[employed, "paid_hourly"].mean())),
        ("weighted_mean_weekly_earnings_employed", _weighted_mean(df.loc[employed, "weekly_earnings_real"], weights.loc[employed])),
        ("weighted_mean_hourly_wage_employed", _weighted_mean(df.loc[employed, "hourly_wage_real"], weights.loc[employed])),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def _format_metric(summary: pd.DataFrame, metric: str, digits: int = 4) -> str:
    value = summary.loc[summary["metric"] == metric, "value"].iloc[0]
    if pd.isna(value):
        return "NA"
    if isinstance(value, str):
        return value
    if "share" in metric:
        return f"{float(value) * 100:.2f}%"
    if metric.startswith("calendar_year"):
        return str(int(value))
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def build_report(summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# SIPP Validation",
            "",
            "This report documents the first real-data validation pass for the public-use Census SIPP standardization lane.",
            "",
            "## Snapshot",
            f"- Rows: {_format_metric(summary, 'rows')}",
            f"- Calendar year: {_format_metric(summary, 'calendar_year_min')} to {_format_metric(summary, 'calendar_year_max')}",
            f"- Female share: {_format_metric(summary, 'female_share')}",
            f"- Employed share: {_format_metric(summary, 'employed_share')}",
            f"- Unemployed share: {_format_metric(summary, 'unemployed_share')}",
            f"- Not in labor force share: {_format_metric(summary, 'not_in_labor_force_share')}",
            f"- Missing labor-force status share: {_format_metric(summary, 'missing_labor_force_share')}",
            "",
            "## Coverage",
            f"- Weekly earnings observed: {_format_metric(summary, 'weekly_earnings_nonnull_share')}",
            f"- Hourly wage observed: {_format_metric(summary, 'hourly_wage_nonnull_share')}",
            f"- Usual hours observed: {_format_metric(summary, 'usual_hours_nonnull_share')}",
            f"- Actual hours observed: {_format_metric(summary, 'actual_hours_nonnull_share')}",
            f"- Occupation observed among employed: {_format_metric(summary, 'occupation_nonnull_share_employed')}",
            f"- Industry observed among employed: {_format_metric(summary, 'industry_nonnull_share_employed')}",
            f"- Paid hourly among employed: {_format_metric(summary, 'paid_hourly_share_employed')}",
            "",
            "## Weighted means among employed",
            f"- Weekly earnings: {_format_metric(summary, 'weighted_mean_weekly_earnings_employed', digits=2)}",
            f"- Hourly wage: {_format_metric(summary, 'weighted_mean_hourly_wage_employed', digits=2)}",
            "",
            "## Interpretation",
            "- The SIPP lane is now validated on a real public-use file rather than only synthetic tests.",
            "- This is still a standardization and validation surface, not yet a full SIPP wage-model integration.",
            "- If future SIPP releases change field names again, the main risk is alias coverage rather than file access.",
        ]
    )


def main(
    input_path: Path = DEFAULT_INPUT,
    report_path: Path = DEFAULT_REPORT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> None:
    df = pd.read_parquet(input_path)
    summary = build_summary(df)
    report = build_report(summary)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
