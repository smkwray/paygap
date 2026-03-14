#!/usr/bin/env python3
"""Build a subgroup-oriented SCE chart-data analysis."""

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
CHART_FILE = RAW_DIR / "sce_labor_chart_data_public.xlsx"

MEASURE_LABELS = {
    "Job Offer Wage Expectations": "expected_offer_wage",
    "Reservation Wage": "reservation_wage",
}
SUBGROUPS = [
    "Mean",
    "<=45 years",
    ">45 years",
    "Less than college degree",
    "College degree or higher",
    "Women",
    "Men",
    "<=$60K",
    ">$60K",
]


def _download_if_missing(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    resp = httpx.get(url, follow_redirects=True, timeout=300)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def stage_public_sce_files() -> Path:
    return _download_if_missing(SCE_CHART_URL, CHART_FILE)


def extract_subgroup_series(chart_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(chart_path, sheet_name="Data", header=None)
    header = raw.iloc[:6].ffill(axis=1)

    selected: dict[int, tuple[str, str]] = {}
    for idx in range(raw.shape[1]):
        measure = header.iat[2, idx]
        subgroup = header.iat[5, idx]
        if measure in MEASURE_LABELS and subgroup in SUBGROUPS:
            selected[idx] = (MEASURE_LABELS[measure], subgroup)

    data = raw.iloc[6:, [0] + list(selected.keys())].copy()
    data.columns = ["date_label"] + [selected[i] for i in selected]
    data = data[data["date_label"].notna()].copy()
    data["date"] = pd.to_datetime(data["date_label"], format="%b %Y")

    for col in data.columns:
        if col in {"date_label", "date"}:
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    tidy = data.melt(id_vars=["date_label", "date"], var_name="series_key", value_name="value")
    tidy[["measure", "subgroup"]] = pd.DataFrame(tidy["series_key"].tolist(), index=tidy.index)
    tidy = tidy.drop(columns=["series_key"])
    tidy = tidy.dropna(subset=["value"]).sort_values(["measure", "subgroup", "date"]).reset_index(drop=True)
    return tidy


def summarize_subgroups(tidy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_rows: list[dict[str, object]] = []
    spread_rows: list[dict[str, object]] = []

    for measure, frame in tidy.groupby("measure"):
        wide = frame.pivot(index="date", columns="subgroup", values="value").sort_index()
        latest = wide.iloc[-1]
        trailing = wide.tail(3).mean()

        for subgroup in SUBGROUPS:
            if subgroup not in wide.columns:
                continue
            latest_rows.append(
                {
                    "measure": measure,
                    "subgroup": subgroup,
                    "latest_date": wide.index[-1].strftime("%Y-%m"),
                    "latest_value": float(latest[subgroup]),
                    "trailing3_mean": float(trailing[subgroup]),
                }
            )

        spreads = {
            "gender_gap_men_minus_women": ("Men", "Women"),
            "education_gap_college_minus_less_than_college": ("College degree or higher", "Less than college degree"),
            "income_gap_high_minus_low": (">$60K", "<=$60K"),
            "age_gap_younger_minus_older": ("<=45 years", ">45 years"),
        }
        for label, (hi, lo) in spreads.items():
            if hi not in wide.columns or lo not in wide.columns:
                continue
            spread_series = wide[hi] - wide[lo]
            spread_rows.append(
                {
                    "measure": measure,
                    "spread": label,
                    "latest_date": wide.index[-1].strftime("%Y-%m"),
                    "latest_value": float(spread_series.iloc[-1]),
                    "trailing3_mean": float(spread_series.tail(3).mean()),
                    "mean_full_sample": float(spread_series.mean()),
                    "share_waves_positive": float((spread_series > 0).mean()),
                }
            )

    return pd.DataFrame(latest_rows), pd.DataFrame(spread_rows)


def write_report(latest_df: pd.DataFrame, spread_df: pd.DataFrame, output_path: Path) -> Path:
    def latest(measure: str, subgroup: str) -> float:
        return float(
            latest_df.loc[
                (latest_df["measure"] == measure) & (latest_df["subgroup"] == subgroup),
                "latest_value",
            ].iloc[0]
        )

    def spread(measure: str, spread_name: str, column: str = "latest_value") -> float:
        return float(
            spread_df.loc[
                (spread_df["measure"] == measure) & (spread_df["spread"] == spread_name),
                column,
            ].iloc[0]
        )

    lines = [
        "# SCE Subgroup Analysis",
        "",
        "This note extends the earlier SCE headline series into subgroup comparisons using the official New York Fed public chart-data workbook.",
        "It still remains mechanism evidence rather than a mergeable control set for ACS/CPS/SIPP.",
        "",
        "## Main findings",
        "",
        f"- Latest expected-offer gender gap: {spread('expected_offer_wage', 'gender_gap_men_minus_women'):.2f}",
        f"- Latest reservation-wage gender gap: {spread('reservation_wage', 'gender_gap_men_minus_women'):.2f}",
        f"- Latest expected-offer education gap: {spread('expected_offer_wage', 'education_gap_college_minus_less_than_college'):.2f}",
        f"- Latest reservation-wage education gap: {spread('reservation_wage', 'education_gap_college_minus_less_than_college'):.2f}",
        f"- Latest expected-offer income gap: {spread('expected_offer_wage', 'income_gap_high_minus_low'):.2f}",
        f"- Latest reservation-wage income gap: {spread('reservation_wage', 'income_gap_high_minus_low'):.2f}",
        "",
        "## Interpretation",
        "",
        "- The gender gap in expected offers and reservation wages is persistent and positive: men are above women in every wave of both public series in this workbook.",
        "- In the latest data, the education and income spreads are larger than the gender spread. That means SCE is showing gender differences inside a broader expectations gradient by schooling and prior income.",
        "- The age spread is smaller than the education and income spreads in both measures.",
        "- This supports a more careful bargaining story: sex differences in outside options and ask wages exist, but they coexist with substantial differences by education and income status.",
        "",
        "## Latest subgroup snapshot",
        "",
        "| Measure | Women | Men | <=45 | >45 | Less than college | College+ | <=$60K | >$60K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Expected offer wage | {latest('expected_offer_wage', 'Women'):.2f} | {latest('expected_offer_wage', 'Men'):.2f} | {latest('expected_offer_wage', '<=45 years'):.2f} | {latest('expected_offer_wage', '>45 years'):.2f} | {latest('expected_offer_wage', 'Less than college degree'):.2f} | {latest('expected_offer_wage', 'College degree or higher'):.2f} | {latest('expected_offer_wage', '<=$60K'):.2f} | {latest('expected_offer_wage', '>$60K'):.2f} |",
        f"| Reservation wage | {latest('reservation_wage', 'Women'):.2f} | {latest('reservation_wage', 'Men'):.2f} | {latest('reservation_wage', '<=45 years'):.2f} | {latest('reservation_wage', '>45 years'):.2f} | {latest('reservation_wage', 'Less than college degree'):.2f} | {latest('reservation_wage', 'College degree or higher'):.2f} | {latest('reservation_wage', '<=$60K'):.2f} | {latest('reservation_wage', '>$60K'):.2f} |",
        "",
        "## Gap persistence",
        "",
        f"- Share of waves with men above women in expected offer wage: {spread('expected_offer_wage', 'gender_gap_men_minus_women', 'share_waves_positive'):.2%}",
        f"- Share of waves with men above women in reservation wage: {spread('reservation_wage', 'gender_gap_men_minus_women', 'share_waves_positive'):.2%}",
        f"- Share of waves with college above less-than-college in expected offer wage: {spread('expected_offer_wage', 'education_gap_college_minus_less_than_college', 'share_waves_positive'):.2%}",
        f"- Share of waves with >$60K above <=$60K in reservation wage: {spread('reservation_wage', 'income_gap_high_minus_low', 'share_waves_positive'):.2%}",
        "",
        "## Bottom line",
        "",
        "The stronger SCE reading is not just that women report lower expected offers and lower reservation wages than men. It is that those sex differences are stable over time and sit inside even larger education and income gradients. That makes SCE useful as evidence on bargaining thresholds and outside options, while still cautioning against oversimplifying the mechanism to gender alone.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stage_public_sce_files()
    tidy = extract_subgroup_series(CHART_FILE)
    latest_df, spread_df = summarize_subgroups(tidy)

    latest_df.to_csv(RESULTS_DIR / "sce_subgroup_latest.csv", index=False)
    spread_df.to_csv(RESULTS_DIR / "sce_subgroup_spreads.csv", index=False)
    write_report(latest_df, spread_df, REPORTS_DIR / "sce_subgroup_analysis.md")

    print(f"Wrote {RESULTS_DIR / 'sce_subgroup_latest.csv'}")
    print(f"Wrote {RESULTS_DIR / 'sce_subgroup_spreads.csv'}")
    print(f"Wrote {REPORTS_DIR / 'sce_subgroup_analysis.md'}")


if __name__ == "__main__":
    main()
