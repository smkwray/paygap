#!/usr/bin/env python3
"""Build a compact cross-dataset synthesis report from canonical outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_summary() -> pd.DataFrame:
    acs_raw = pd.read_csv(RESULTS_DIR / "trends" / "acs_raw_gap_trend.csv")
    acs_ols = pd.read_csv(RESULTS_DIR / "trends" / "acs_ols_trend.csv")
    acs_unc = pd.read_csv(RESULTS_DIR / "diagnostics" / "acs_uncertainty_summary.csv")
    cps_raw = pd.read_csv(RESULTS_DIR / "trends" / "cps_asec_raw_gap_trend.csv")
    cps_ols = pd.read_csv(RESULTS_DIR / "trends" / "cps_asec_ols_trend.csv")
    sipp_raw = pd.read_csv(RESULTS_DIR / "sipp" / "2023" / "raw_gap.csv")
    sipp_ols = pd.read_csv(RESULTS_DIR / "sipp" / "2023" / "ols_sequential.csv")
    atus = pd.read_csv(RESULTS_DIR / "atus" / "time_use_by_gender.csv")
    sce = pd.read_csv(RESULTS_DIR / "diagnostics" / "sce_public_summary_metrics.csv")

    acs_latest_raw = acs_raw.sort_values("year").iloc[-1]
    acs_latest_m5 = acs_ols.loc[acs_ols["model"] == "M5"].sort_values("year").iloc[-1]
    acs_latest_unc = acs_unc.loc[(acs_unc["year"] == int(acs_latest_raw["year"])) & (acs_unc["metric"] == "raw_gap_pct")].iloc[0]
    cps_latest_raw = cps_raw.sort_values("year").iloc[-1]
    cps_latest_full = cps_ols.loc[cps_ols["model"] == "M_full"].sort_values("year").iloc[-1]
    sipp_latest_s3 = sipp_ols.loc[sipp_ols["model"] == "SIPP3"].iloc[0]

    def sce_metric(name: str) -> float | str:
        return sce.loc[sce["metric"] == name, "value"].iloc[0]

    def atus_gap(activity: str) -> float:
        return float(atus.loc[atus["activity"] == activity, "gap_minutes"].iloc[0])

    rows = [
        ("acs_2023_raw_gap_pct", float(acs_latest_raw["gap_pct"]), "ACS 2023 raw hourly gap"),
        ("acs_2023_m5_pct_gap", float(abs(acs_latest_m5["pct_gap"])), "ACS 2023 adjusted hourly gap from M5"),
        ("acs_2023_raw_gap_ci90_low", float(acs_latest_unc["ci90_low"]), "ACS 2023 raw-gap 90% CI lower bound"),
        ("acs_2023_raw_gap_ci90_high", float(acs_latest_unc["ci90_high"]), "ACS 2023 raw-gap 90% CI upper bound"),
        ("cps_2023_raw_gap_pct", float(cps_latest_raw["gap_pct"]), "CPS ASEC 2023 raw hourly gap"),
        ("cps_2023_m_full_pct_gap", float(abs(cps_latest_full["pct_gap"])), "CPS ASEC 2023 adjusted hourly gap"),
        ("sipp_2023_raw_gap_pct", float(sipp_raw.iloc[0]["gap_pct"]), "SIPP 2023 raw hourly gap"),
        ("sipp_2023_sipp3_pct_gap", float(abs(sipp_latest_s3["pct_gap"])), "SIPP 2023 adjusted hourly gap from SIPP3"),
        ("atus_paid_work_gap_minutes", atus_gap("minutes_paid_work_diary"), "Female-minus-male paid-work minutes/day"),
        ("atus_housework_gap_minutes", atus_gap("minutes_housework"), "Female-minus-male housework minutes/day"),
        ("atus_childcare_gap_minutes", atus_gap("minutes_childcare"), "Female-minus-male childcare minutes/day"),
        (
            "sce_latest_expected_offer_gap",
            float(sce_metric("latest_expected_offer_wage_gap_men_minus_women")),
            "Latest men-minus-women expected offer wage gap in SCE public series",
        ),
        (
            "sce_latest_reservation_gap",
            float(sce_metric("latest_reservation_wage_gap_men_minus_women")),
            "Latest men-minus-women reservation wage gap in SCE public series",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


def build_report(summary: pd.DataFrame) -> str:
    metric = summary.set_index("metric")["value"]
    return "\n".join(
        [
            "# Final Synthesis",
            "",
            "This report condenses the repo's canonical cross-dataset findings into one closing surface.",
            "",
            "## Headline cross-dataset results",
            "",
            f"- ACS 2023 raw hourly gap: {metric['acs_2023_raw_gap_pct']:.2f}% "
            f"(90% CI {metric['acs_2023_raw_gap_ci90_low']:.2f}% to {metric['acs_2023_raw_gap_ci90_high']:.2f}%)",
            f"- ACS 2023 adjusted hourly gap (`M5`): {metric['acs_2023_m5_pct_gap']:.2f}%",
            f"- CPS ASEC 2023 raw hourly gap: {metric['cps_2023_raw_gap_pct']:.2f}%",
            f"- CPS ASEC 2023 adjusted hourly gap (`M_full`): {metric['cps_2023_m_full_pct_gap']:.2f}%",
            f"- SIPP 2023 raw hourly gap: {metric['sipp_2023_raw_gap_pct']:.2f}%",
            f"- SIPP 2023 adjusted hourly gap (`SIPP3`): {metric['sipp_2023_sipp3_pct_gap']:.2f}%",
            "",
            "## Mechanism evidence",
            "",
            f"- ATUS paid-work gap: {metric['atus_paid_work_gap_minutes']:.2f} minutes/day",
            f"- ATUS housework gap: {metric['atus_housework_gap_minutes']:.2f} minutes/day",
            f"- ATUS childcare gap: {metric['atus_childcare_gap_minutes']:.2f} minutes/day",
            f"- SCE latest expected-offer gap: {metric['sce_latest_expected_offer_gap']:.2f}",
            f"- SCE latest reservation-wage gap: {metric['sce_latest_reservation_gap']:.2f}",
            "",
            "## Interpretation",
            "",
            "- ACS, CPS, and SIPP all point in the same direction: the worker hourly gap is substantial before controls and remains material after the controls each dataset can honestly support.",
            "- ACS remains the primary headline source because it has the richest repeated-year surface and survey-consistent uncertainty.",
            "- CPS and SIPP act as external checks rather than exact replicas of the ACS estimand.",
            "- ATUS and SCE strengthen interpretation: time-allocation burdens and expectations/outside-options move in directions consistent with part of the observed gap, but they do not eliminate the residual worker-gap result.",
            "",
            "## Bottom line",
            "",
            "Across the public-data sources in this repo, the gender pay gap is not a single-number claim. The realized gap is large, it narrows after controls, but it does not disappear in ACS, CPS, or the first SIPP model surface. Mechanism evidence from ATUS and SCE is directionally consistent with family, schedule, and bargaining-related channels, while still leaving a meaningful residual worker gap in the main wage datasets.",
        ]
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "diagnostics").mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    summary.to_csv(RESULTS_DIR / "diagnostics" / "final_synthesis_metrics.csv", index=False)
    (REPORTS_DIR / "final_synthesis.md").write_text(build_report(summary) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'diagnostics' / 'final_synthesis_metrics.csv'}")
    print(f"Wrote {REPORTS_DIR / 'final_synthesis.md'}")


if __name__ == "__main__":
    main()
