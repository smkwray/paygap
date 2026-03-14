#!/usr/bin/env python3
"""Build a sensitivity summary for ACS Oaxaca instability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DIAG_DIR = RESULTS_DIR / "diagnostics"


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_oaxaca_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    oaxaca = _load_csv(RESULTS_DIR / "trends" / "acs_oaxaca_trend.csv")
    ols = _load_csv(RESULTS_DIR / "trends" / "acs_ols_trend.csv")
    m5 = ols.loc[ols["model"] == "M5", ["year", "female_coef", "pct_gap"]].copy()
    merged = oaxaca.merge(m5, on="year", how="left")

    pre = merged.loc[merged["year"] <= 2018].copy()
    post = merged.loc[merged["year"] >= 2019].copy()

    pre_total = pre["total_gap"].mean()
    pre_explained = pre["explained"].mean()
    pre_unexplained = pre["unexplained"].mean()
    pre_unexplained_pct = pre["unexplained_pct"].mean()

    post_total = post["total_gap"].mean()
    post_explained = post["explained"].mean()
    post_unexplained = post["unexplained"].mean()
    post_unexplained_pct = post["unexplained_pct"].mean()

    merged["pre_period_total_gap_mean"] = pre_total
    merged["pre_period_explained_mean"] = pre_explained
    merged["pre_period_unexplained_mean"] = pre_unexplained
    merged["total_gap_delta_vs_pre_mean"] = merged["total_gap"] - pre_total
    merged["explained_delta_vs_pre_mean"] = merged["explained"] - pre_explained
    merged["unexplained_delta_vs_pre_mean"] = merged["unexplained"] - pre_unexplained
    merged["unexplained_pct_delta_vs_pre_mean"] = merged["unexplained_pct"] - pre_unexplained_pct

    summary = pd.DataFrame([
        {
            "metric": "pre_2019_total_gap_mean",
            "value": pre_total,
            "note": "Average ACS Oaxaca total log gap for 2015-2018.",
        },
        {
            "metric": "post_2019_total_gap_mean",
            "value": post_total,
            "note": "Average ACS Oaxaca total log gap for 2019-2023.",
        },
        {
            "metric": "pre_2019_explained_mean",
            "value": pre_explained,
            "note": "Average ACS Oaxaca explained component for 2015-2018.",
        },
        {
            "metric": "post_2019_explained_mean",
            "value": post_explained,
            "note": "Average ACS Oaxaca explained component for 2019-2023.",
        },
        {
            "metric": "pre_2019_unexplained_mean",
            "value": pre_unexplained,
            "note": "Average ACS Oaxaca unexplained component for 2015-2018.",
        },
        {
            "metric": "post_2019_unexplained_mean",
            "value": post_unexplained,
            "note": "Average ACS Oaxaca unexplained component for 2019-2023.",
        },
        {
            "metric": "explained_component_change_post_minus_pre",
            "value": post_explained - pre_explained,
            "note": "Post-2019 average explained component minus pre-2019 average.",
        },
        {
            "metric": "unexplained_component_change_post_minus_pre",
            "value": post_unexplained - pre_unexplained,
            "note": "Post-2019 average unexplained component minus pre-2019 average.",
        },
        {
            "metric": "unexplained_share_change_pp_post_minus_pre",
            "value": post_unexplained_pct - pre_unexplained_pct,
            "note": "Change in average unexplained share in percentage points.",
        },
        {
            "metric": "post_share_if_pre_explained_mean_held",
            "value": (post_total - pre_explained) / post_total * 100,
            "note": "Counterfactual post-2019 unexplained share if the explained component had stayed at its pre-2019 mean.",
        },
        {
            "metric": "post_share_if_pre_unexplained_mean_held",
            "value": pre_unexplained / post_total * 100,
            "note": "Counterfactual post-2019 unexplained share if the unexplained component had stayed at its pre-2019 mean.",
        },
        {
            "metric": "max_unexplained_pct_year",
            "value": float(merged.loc[merged["unexplained_pct"].idxmax(), "year"]),
            "note": "Year with the highest unexplained share.",
        },
    ])

    return merged, summary


def write_report(yearly: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    def metric(name: str) -> float:
        return float(summary.loc[summary["metric"] == name, "value"].iloc[0])

    lines = [
        "# Oaxaca Sensitivity Note",
        "",
        "This note isolates what changed in the ACS Oaxaca decomposition after 2019.",
        "",
        "## Main finding",
        "",
        f"- The average unexplained share rises from {metric('pre_2019_unexplained_mean') / metric('pre_2019_total_gap_mean') * 100:.2f}% in 2015-2018 to {metric('post_2019_unexplained_mean') / metric('post_2019_total_gap_mean') * 100:.2f}% in 2019-2023.",
        f"- The total log gap changes only modestly: {metric('pre_2019_total_gap_mean'):.4f} to {metric('post_2019_total_gap_mean'):.4f}.",
        f"- The explained component falls much more sharply: {metric('pre_2019_explained_mean'):.4f} to {metric('post_2019_explained_mean'):.4f}.",
        f"- The unexplained component does rise, but more modestly in level terms: {metric('pre_2019_unexplained_mean'):.4f} to {metric('post_2019_unexplained_mean'):.4f}.",
        "",
        "## Interpretation",
        "",
        "- The post-2019 jump in unexplained share is driven more by the explained component collapsing than by the total wage gap blowing out.",
        "- That is why the unexplained share looks unstable while the year-by-year ACS M5 gap remains in a comparatively tight band.",
        "- In other words, the Oaxaca percentage shares are more fragile than the core sequential OLS trend.",
        "",
        "## Counterfactual checks",
        "",
        f"- If the explained component had stayed at its 2015-2018 mean, the post-2019 unexplained share would be about {metric('post_share_if_pre_explained_mean_held'):.2f}% instead of {metric('unexplained_share_change_pp_post_minus_pre') + (metric('pre_2019_unexplained_mean') / metric('pre_2019_total_gap_mean') * 100):.2f}%.",
        f"- If the unexplained component had stayed at its 2015-2018 mean, the post-2019 unexplained share would be about {metric('post_share_if_pre_unexplained_mean_held'):.2f}%.",
        "- That pattern reinforces the same reading: the share jump is mostly about a shrinking explained component under a fairly stable total gap.",
        "",
        "## Year-level table",
        "",
        "| Year | Total gap | Explained | Unexplained | Unexplained % | M5 coef | M5 gap % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in yearly.itertuples(index=False):
        lines.append(
            f"| {int(row.year)} | {row.total_gap:.4f} | {row.explained:.4f} | {row.unexplained:.4f} | "
            f"{row.unexplained_pct:.2f} | {row.female_coef:.4f} | {row.pct_gap:.2f} |"
        )

    lines.extend([
        "",
        "## Practical conclusion",
        "",
        "Keep Oaxaca as a supplemental decomposition result. The year-by-year ACS raw-gap and M5 trend remain the stronger headline series because they stay stable while the decomposition share breakdown becomes sensitive after 2019.",
    ])
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    yearly, summary = build_oaxaca_sensitivity()
    yearly.to_csv(DIAG_DIR / "acs_oaxaca_sensitivity_yearly.csv", index=False)
    summary.to_csv(DIAG_DIR / "acs_oaxaca_sensitivity_summary.csv", index=False)
    write_report(yearly, summary, REPORTS_DIR / "oaxaca_sensitivity.md")
    print(f"Wrote {DIAG_DIR / 'acs_oaxaca_sensitivity_yearly.csv'}")
    print(f"Wrote {DIAG_DIR / 'acs_oaxaca_sensitivity_summary.csv'}")
    print(f"Wrote {REPORTS_DIR / 'oaxaca_sensitivity.md'}")


if __name__ == "__main__":
    main()
