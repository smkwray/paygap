#!/usr/bin/env python3
"""Build an empirical SCE public-series note from official New York Fed files."""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sce_labor_market"
RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics"
REPORTS_DIR = PROJECT_ROOT / "reports"

SCE_CHART_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/microeconomics/"
    "interactive/downloads/sce-labor-chart-data-public.xlsx?sc_lang=en"
)
SCE_MICRODATA_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/microeconomics/"
    "interactive/downloads/sce-labor-microdata-public.xlsx?sc_lang=en"
)
CHART_FILE = RAW_DIR / "sce_labor_chart_data_public.xlsx"
MICRODATA_FILE = RAW_DIR / "sce_labor_microdata_public.xlsx"


def _download_if_missing(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    resp = httpx.get(url, follow_redirects=True, timeout=300)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def stage_public_sce_files() -> dict[str, Path]:
    return {
        "chart_data": _download_if_missing(SCE_CHART_URL, CHART_FILE),
        "microdata": _download_if_missing(SCE_MICRODATA_URL, MICRODATA_FILE),
    }


def extract_public_series(chart_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(chart_path, sheet_name="Data", header=None)
    data = raw.iloc[7:].copy()
    cols = [0, 176, 182, 183, 186, 192, 193]
    series = data[cols].copy()
    series.columns = [
        "date_label",
        "expected_offer_wage_mean",
        "expected_offer_wage_women",
        "expected_offer_wage_men",
        "reservation_wage_mean",
        "reservation_wage_women",
        "reservation_wage_men",
    ]
    series = series[series["date_label"].notna()].copy()
    series["date"] = pd.to_datetime(series["date_label"], format="%b %Y")
    numeric_cols = [c for c in series.columns if c not in {"date_label", "date"}]
    for col in numeric_cols:
        series[col] = pd.to_numeric(series[col], errors="coerce")

    series["expected_offer_wage_gap_men_minus_women"] = (
        series["expected_offer_wage_men"] - series["expected_offer_wage_women"]
    )
    series["reservation_wage_gap_men_minus_women"] = (
        series["reservation_wage_men"] - series["reservation_wage_women"]
    )
    return series.sort_values("date").reset_index(drop=True)


def summarize_public_series(series: pd.DataFrame) -> pd.DataFrame:
    latest = series.iloc[-1]
    trailing = series.tail(3)
    earliest = series.iloc[0]

    rows = [
        {
            "metric": "latest_date",
            "value": latest["date"].strftime("%Y-%m"),
            "note": "Latest date present in the official public chart-data workbook.",
        },
        {
            "metric": "latest_expected_offer_wage_women",
            "value": latest["expected_offer_wage_women"],
            "note": "Latest women expected offer wage.",
        },
        {
            "metric": "latest_expected_offer_wage_men",
            "value": latest["expected_offer_wage_men"],
            "note": "Latest men expected offer wage.",
        },
        {
            "metric": "latest_expected_offer_wage_gap_men_minus_women",
            "value": latest["expected_offer_wage_gap_men_minus_women"],
            "note": "Latest men-minus-women expected offer wage gap.",
        },
        {
            "metric": "latest_reservation_wage_women",
            "value": latest["reservation_wage_women"],
            "note": "Latest women reservation wage.",
        },
        {
            "metric": "latest_reservation_wage_men",
            "value": latest["reservation_wage_men"],
            "note": "Latest men reservation wage.",
        },
        {
            "metric": "latest_reservation_wage_gap_men_minus_women",
            "value": latest["reservation_wage_gap_men_minus_women"],
            "note": "Latest men-minus-women reservation wage gap.",
        },
        {
            "metric": "trailing_year_expected_offer_gap_mean",
            "value": trailing["expected_offer_wage_gap_men_minus_women"].mean(),
            "note": "Average men-minus-women expected offer wage gap over the latest three waves.",
        },
        {
            "metric": "trailing_year_reservation_gap_mean",
            "value": trailing["reservation_wage_gap_men_minus_women"].mean(),
            "note": "Average men-minus-women reservation wage gap over the latest three waves.",
        },
        {
            "metric": "reservation_wage_gap_change_since_start",
            "value": latest["reservation_wage_gap_men_minus_women"] - earliest["reservation_wage_gap_men_minus_women"],
            "note": "Change in men-minus-women reservation wage gap from first to latest wave.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(series: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    def metric(name: str):
        return summary.loc[summary["metric"] == name, "value"].iloc[0]

    latest_date = str(metric("latest_date"))
    lines = [
        "# SCE Public-Series Empirical Note",
        "",
        "This note uses the official New York Fed public chart-data workbook for the Labor Market Survey.",
        "It provides an empirical SCE evidence layer without attempting a person-level merge onto ACS/CPS.",
        "",
        "## Data used",
        "",
        f"- Chart-data workbook: `{CHART_FILE.relative_to(PROJECT_ROOT)}`",
        f"- Public microdata workbook staged for future work: `{MICRODATA_FILE.relative_to(PROJECT_ROOT)}`",
        f"- Latest public chart wave in the workbook: {latest_date}",
        "",
        "## Main findings",
        "",
        f"- Latest expected offer wage, women: {float(metric('latest_expected_offer_wage_women')):.2f}",
        f"- Latest expected offer wage, men: {float(metric('latest_expected_offer_wage_men')):.2f}",
        f"- Latest men-minus-women expected offer wage gap: {float(metric('latest_expected_offer_wage_gap_men_minus_women')):.2f}",
        f"- Latest reservation wage, women: {float(metric('latest_reservation_wage_women')):.2f}",
        f"- Latest reservation wage, men: {float(metric('latest_reservation_wage_men')):.2f}",
        f"- Latest men-minus-women reservation wage gap: {float(metric('latest_reservation_wage_gap_men_minus_women')):.2f}",
        "",
        f"- Average men-minus-women expected offer wage gap over the latest three waves: {float(metric('trailing_year_expected_offer_gap_mean')):.2f}",
        f"- Average men-minus-women reservation wage gap over the latest three waves: {float(metric('trailing_year_reservation_gap_mean')):.2f}",
        "",
        "## Interpretation",
        "",
        "- The official public SCE series show that men report higher expected offer wages and higher reservation wages than women in the latest available waves.",
        "- That is directly relevant to the bargaining/expectations question: the difference shows up before any attempted person-level merge to ACS/CPS.",
        "- This still does not become a control variable in the main wage regressions. It remains supporting evidence on expectations and outside options.",
        "",
        "## Practical use in this repo",
        "",
        "- Use ACS/CPS/ATUS as the main realized-gap evidence.",
        "- Use this SCE public-series note to support claims about expected offers and reservation wages.",
        "- If deeper SCE work is wanted later, the staged public microdata workbook can be used for more detailed within-SCE analysis if the needed subgroup variables are available or can be mapped cleanly.",
        "",
        "## Latest series snapshot",
        "",
        "| Date | Women expected offer | Men expected offer | Women reservation wage | Men reservation wage |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in series.tail(6).itertuples(index=False):
        lines.append(
            f"| {row.date.strftime('%Y-%m')} | {row.expected_offer_wage_women:.2f} | {row.expected_offer_wage_men:.2f} | "
            f"{row.reservation_wage_women:.2f} | {row.reservation_wage_men:.2f} |"
        )

    lines.extend([
        "",
        "## Bottom line",
        "",
        "The SCE public series now give this project a direct empirical expectations layer: women report lower reservation wages and lower expected offer wages than men in the latest official New York Fed public data. That supports a bargaining and outside-options channel, but it does not displace the realized worker-gap findings from ACS and CPS.",
    ])
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stage_public_sce_files()
    series = extract_public_series(CHART_FILE)
    summary = summarize_public_series(series)

    series.to_csv(RESULTS_DIR / "sce_public_series.csv", index=False)
    summary.to_csv(RESULTS_DIR / "sce_public_summary_metrics.csv", index=False)
    write_report(series, summary, REPORTS_DIR / "sce_public_analysis.md")

    print(f"Wrote {RESULTS_DIR / 'sce_public_series.csv'}")
    print(f"Wrote {RESULTS_DIR / 'sce_public_summary_metrics.csv'}")
    print(f"Wrote {REPORTS_DIR / 'sce_public_analysis.md'}")


if __name__ == "__main__":
    main()
