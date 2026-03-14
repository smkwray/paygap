#!/usr/bin/env python3
"""Build a consolidated robustness appendix from completed artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DIAG_DIR = RESULTS_DIR / "diagnostics"


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_summary_rows() -> pd.DataFrame:
    uncertainty = _load_csv(DIAG_DIR / "acs_uncertainty_summary.csv")
    family = _load_csv(DIAG_DIR / "acs_family_rebuild_comparison.csv")
    acs_selection = _load_csv(RESULTS_DIR / "trends" / "acs_selection_trend.csv")
    cps_selection = _load_csv(RESULTS_DIR / "trends" / "cps_selection_trend.csv")
    context_status = _load_csv(DIAG_DIR / "context_data_status.csv")
    raw_uncertainty = _load_csv(RESULTS_DIR / "trends" / "acs_raw_gap_trend_with_uncertainty.csv")
    oaxaca = _load_csv(RESULTS_DIR / "trends" / "acs_oaxaca_trend.csv")

    m5_uncertainty = uncertainty.loc[uncertainty["metric"] == "M5_female_coef"].copy()
    family_year = family.loc[family["scope"] == "year"].copy()
    family_pooled = family.loc[family["scope"] == "pooled"].copy()
    acs_s2 = acs_selection.loc[acs_selection["model"] == "S2"].copy()
    cps_s2 = cps_selection.loc[cps_selection["model"] == "S2"].copy()
    pre_2019 = oaxaca.loc[oaxaca["year"] <= 2018, "unexplained_pct"]
    post_2019 = oaxaca.loc[oaxaca["year"] >= 2019, "unexplained_pct"]

    rows = [
        {
            "section": "acs_uncertainty",
            "metric": "raw_gap_avg_moe90_pp",
            "value": raw_uncertainty["moe90"].mean(),
            "note": "Average ACS raw-gap 90% margin of error in percentage points.",
        },
        {
            "section": "acs_uncertainty",
            "metric": "m5_avg_se",
            "value": m5_uncertainty["se"].mean(),
            "note": "Average ACS M5 SDR standard error.",
        },
        {
            "section": "acs_family_rebuild",
            "metric": "mean_m5_delta",
            "value": family_year["m5_delta"].mean(),
            "note": "Average rebuilt minus old ACS year-level M5 coefficient.",
        },
        {
            "section": "acs_family_rebuild",
            "metric": "max_abs_m5_delta",
            "value": family_year["m5_delta"].abs().max(),
            "note": "Largest absolute year-level M5 change from correcting ACS family fields.",
        },
        {
            "section": "acs_family_rebuild",
            "metric": "pooled_p5_delta",
            "value": family_pooled["m5_delta"].iloc[0] if not family_pooled.empty else pd.NA,
            "note": "Rebuilt minus old pooled ACS P5 coefficient.",
        },
        {
            "section": "acs_selection",
            "metric": "s2_expected_gap_pct_mean",
            "value": acs_s2["combined_expected_earnings_gap_pct"].mean(),
            "note": "Mean ACS S2 combined expected annual-earnings gap.",
        },
        {
            "section": "acs_selection",
            "metric": "s2_ipw_worker_gap_pct_mean",
            "value": acs_s2["ipw_worker_hourly_gap_pct"].mean(),
            "note": "Mean ACS S2 IPW worker hourly wage gap.",
        },
        {
            "section": "cps_selection",
            "metric": "s2_expected_gap_pct_mean",
            "value": cps_s2["combined_expected_earnings_gap_pct"].mean(),
            "note": "Mean CPS S2 combined expected annual-earnings gap.",
        },
        {
            "section": "cps_selection",
            "metric": "s2_ipw_worker_gap_pct_mean",
            "value": cps_s2["ipw_worker_hourly_gap_pct"].mean(),
            "note": "Mean CPS S2 IPW worker hourly wage gap.",
        },
        {
            "section": "oaxaca",
            "metric": "pre_2019_unexplained_pct_mean",
            "value": pre_2019.mean(),
            "note": "Average ACS Oaxaca unexplained share for 2015-2018.",
        },
        {
            "section": "oaxaca",
            "metric": "post_2019_unexplained_pct_mean",
            "value": post_2019.mean(),
            "note": "Average ACS Oaxaca unexplained share for 2019-2023.",
        },
        {
            "section": "context",
            "metric": "present_sources_count",
            "value": int((context_status["status"] == "present").sum()),
            "note": "Count of context sources currently cached and available.",
        },
        {
            "section": "context",
            "metric": "manual_staging_sources_count",
            "value": int((context_status["status"] == "staged_manual").sum()),
            "note": "Count of context sources that currently require manual staging.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, path: Path) -> Path:
    def metric(section: str, name: str) -> float:
        return float(summary.loc[(summary["section"] == section) & (summary["metric"] == name), "value"].iloc[0])

    lines = [
        "# Robustness Appendix",
        "",
        "This appendix consolidates the completed robustness surfaces already built in the repo.",
        "It is not a new model family; it is a summary layer over the uncertainty, sensitivity, selection, decomposition, and context artifacts.",
        "",
        "## ACS Survey Uncertainty",
        "",
        f"- ACS raw-gap 90% margins of error average {metric('acs_uncertainty', 'raw_gap_avg_moe90_pp'):.2f} percentage points across the full series.",
        f"- ACS M5 SDR standard errors average {metric('acs_uncertainty', 'm5_avg_se'):.4f}.",
        "- Interpretation: the year-by-year ACS headline series is statistically tight; the main uncertainty in this project is methodological rather than sampling noise.",
        "",
        "## ACS Family-Field Sensitivity",
        "",
        f"- Correcting the ACS family variables shifts the year-level M5 coefficient by {metric('acs_family_rebuild', 'mean_m5_delta'):.4f} on average.",
        f"- The largest absolute year-level M5 change is {metric('acs_family_rebuild', 'max_abs_m5_delta'):.4f}.",
        f"- The pooled ACS P5 coefficient shifts by {metric('acs_family_rebuild', 'pooled_p5_delta'):.4f}.",
        "- Interpretation: the family-field bug was real and worth fixing, but it does not overturn the main residual-gap result.",
        "",
        "## Selection Robustness",
        "",
        f"- ACS `S2` mean combined expected annual-earnings gap: {metric('acs_selection', 's2_expected_gap_pct_mean'):.2f}%",
        f"- ACS `S2` mean IPW worker hourly wage gap: {metric('acs_selection', 's2_ipw_worker_gap_pct_mean'):.2f}%",
        f"- CPS `S2` mean combined expected annual-earnings gap: {metric('cps_selection', 's2_expected_gap_pct_mean'):.2f}%",
        f"- CPS `S2` mean IPW worker hourly wage gap: {metric('cps_selection', 's2_ipw_worker_gap_pct_mean'):.2f}%",
        "- Interpretation: annual-earnings gaps are larger than worker-only hourly wage gaps, but reweighting on employment selection does not collapse the worker-gap result.",
        "",
        "## Oaxaca Stability",
        "",
        f"- Mean unexplained share in 2015-2018: {metric('oaxaca', 'pre_2019_unexplained_pct_mean'):.2f}%",
        f"- Mean unexplained share in 2019-2023: {metric('oaxaca', 'post_2019_unexplained_pct_mean'):.2f}%",
        "- Interpretation: decomposition shares become much less stable after 2019 than the sequential OLS estimates, so Oaxaca should stay secondary to the year-by-year M5 trend in external reporting.",
        "",
        "## Context Controls Status",
        "",
        f"- Cached context sources currently present: {int(metric('context', 'present_sources_count'))}",
        f"- Sources currently requiring manual staging: {int(metric('context', 'manual_staging_sources_count'))}",
        "- Interpretation: BEA RPP and QCEW are available, while OEWS is documented as manual-staging-only from this environment. Context data remain supplemental rather than core blockers.",
        "",
        "## Source Artifacts",
        "",
        "- `results/diagnostics/acs_uncertainty_summary.csv`",
        "- `results/trends/acs_raw_gap_trend_with_uncertainty.csv`",
        "- `results/diagnostics/acs_family_rebuild_comparison.csv`",
        "- `results/trends/acs_selection_trend.csv`",
        "- `results/trends/cps_selection_trend.csv`",
        "- `results/diagnostics/context_data_status.csv`",
        "",
        "## Bottom Line",
        "",
        "Across the completed robustness surfaces, the central result survives: the realized worker earnings gap is persistent, the residual ACS gap remains material after observables, family-field corrections do not eliminate it, and selection sensitivity changes the estimand more than it changes the existence of the worker-gap result.",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary_rows()
    summary.to_csv(DIAG_DIR / "robustness_appendix_summary.csv", index=False)
    write_report(summary, REPORTS_DIR / "robustness_appendix.md")
    print(f"Wrote {DIAG_DIR / 'robustness_appendix_summary.csv'}")
    print(f"Wrote {REPORTS_DIR / 'robustness_appendix.md'}")


if __name__ == "__main__":
    main()
