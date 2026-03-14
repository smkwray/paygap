#!/usr/bin/env python3
"""Build the SCE supplemental report and measure map."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sce_labor_market"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gender_gap.downloaders.sce import (  # noqa: E402
    SCELaborMarketDownloader,
    SCE_CODEBOOK_URL,
    SCE_DATABANK_URL,
    SCE_FAQ_URL,
    SCE_LABOR_URL,
    SCE_MAIN_URL,
    SCE_QUESTIONNAIRE_URL,
    SCE_RESERVATION_WAGE_POST_URL,
)


def build_sce_measure_map() -> pd.DataFrame:
    rows = [
        {
            "construct": "reservation_wage",
            "definition": "Lowest wage or salary the respondent says they would accept for a job they would consider.",
            "why_it_matters": "Closest public survey measure to the wage a worker says they require before accepting a new job.",
            "best_repo_use": "Supporting evidence on bargaining thresholds and outside options.",
            "mergeable_into_acs_cps": False,
            "reason_not_mergeable": "Different survey, different sample, different cadence, and no crosswalk to ACS/CPS person records.",
            "primary_source_url": SCE_QUESTIONNAIRE_URL,
        },
        {
            "construct": "expected_offer_wage",
            "definition": "Expected wage or salary of future offers, conditional on receiving an offer.",
            "why_it_matters": "Captures what workers think the market will pay, which is distinct from realized current wages.",
            "best_repo_use": "Expectation benchmark beside ACS/CPS realized worker gaps.",
            "mergeable_into_acs_cps": False,
            "reason_not_mergeable": "Expectation variable from a separate rotating module, not observed in ACS/CPS wage files.",
            "primary_source_url": SCE_LABOR_URL,
        },
        {
            "construct": "offer_receipt_and_acceptance",
            "definition": "Number of offers received, offer wages, and whether offers were accepted, rejected, or still under consideration.",
            "why_it_matters": "Lets the project discuss realized job-offer dynamics instead of treating bargaining as purely hypothetical.",
            "best_repo_use": "Mechanism evidence on job search and offer conversion.",
            "mergeable_into_acs_cps": False,
            "reason_not_mergeable": "Observed over the last four months in a separate New York Fed panel, not in ACS/CPS.",
            "primary_source_url": SCE_CODEBOOK_URL,
        },
        {
            "construct": "job_offer_expectations",
            "definition": "Percent chance of receiving at least one offer and expected number of offers over the next four months.",
            "why_it_matters": "Measures perceived opportunity set, which can shape ask wages and acceptance thresholds.",
            "best_repo_use": "Context for interpreting gender differences in outside options or bargaining leverage.",
            "mergeable_into_acs_cps": False,
            "reason_not_mergeable": "Forward-looking expectations from a smaller survey, not a realized wage control in the main files.",
            "primary_source_url": SCE_LABOR_URL,
        },
        {
            "construct": "job_search_intensity",
            "definition": "Whether the respondent searched in the last four weeks, search channels used, and hours spent searching.",
            "why_it_matters": "Helps separate bargaining claims from search effort and labor-market attachment.",
            "best_repo_use": "Supporting mechanism evidence next to ACS/CPS selection robustness.",
            "mergeable_into_acs_cps": False,
            "reason_not_mergeable": "Search-intensity module is not available person-for-person in ACS/CPS.",
            "primary_source_url": SCE_QUESTIONNAIRE_URL,
        },
        {
            "construct": "retirement_and_transition_expectations",
            "definition": "Probabilities of future employment states and working beyond ages 62 and 67.",
            "why_it_matters": "Adds forward-looking labor-supply expectations that can differ from realized annual earnings.",
            "best_repo_use": "Supplemental interpretation for extensive-margin and lifecycle channels.",
            "mergeable_into_acs_cps": False,
            "reason_not_mergeable": "Expectation measures live in SCE only and are not aligned to ACS/CPS survey timing.",
            "primary_source_url": SCE_FAQ_URL,
        },
    ]
    return pd.DataFrame(rows)


def collect_repo_context() -> dict[str, float]:
    acs_ols = pd.read_csv(RESULTS_DIR / "trends" / "acs_ols_trend.csv")
    acs_m5 = acs_ols.loc[acs_ols["model"] == "M5"].sort_values("year").copy()

    acs_selection = pd.read_csv(RESULTS_DIR / "trends" / "acs_selection_trend.csv")
    acs_s2 = acs_selection.loc[acs_selection["model"] == "S2"].sort_values("year").copy()

    cps_selection = pd.read_csv(RESULTS_DIR / "trends" / "cps_selection_trend.csv")
    cps_s2 = cps_selection.loc[cps_selection["model"] == "S2"].sort_values("year").copy()

    atus = pd.read_csv(RESULTS_DIR / "atus" / "time_use_by_gender.csv").set_index("activity")

    return {
        "acs_2023_m5_pct_gap": float(abs(acs_m5.loc[acs_m5["year"] == 2023, "pct_gap"].iloc[0])),
        "acs_m5_mean_pct_gap": float(acs_m5["pct_gap"].abs().mean()),
        "acs_s2_expected_gap_pct": float(acs_s2["combined_expected_earnings_gap_pct"].mean()),
        "acs_s2_ipw_gap_pct": float(acs_s2["ipw_worker_hourly_gap_pct"].mean()),
        "cps_s2_expected_gap_pct": float(cps_s2["combined_expected_earnings_gap_pct"].mean()),
        "cps_s2_ipw_gap_pct": float(cps_s2["ipw_worker_hourly_gap_pct"].mean()),
        "atus_housework_gap_minutes": float(atus.loc["minutes_housework", "gap_minutes"]),
        "atus_childcare_gap_minutes": float(atus.loc["minutes_childcare", "gap_minutes"]),
        "atus_paid_work_gap_minutes": float(atus.loc["minutes_paid_work_diary", "gap_minutes"]),
    }


def write_sce_report(metrics: dict[str, float], measure_map: pd.DataFrame, output_path: Path) -> Path:
    measure_lines = []
    for row in measure_map.itertuples(index=False):
        mergeable = "No" if not row.mergeable_into_acs_cps else "Yes"
        measure_lines.extend(
            [
                f"### {row.construct.replace('_', ' ').title()}",
                f"- What it measures: {row.definition}",
                f"- Why it matters: {row.why_it_matters}",
                f"- Best use here: {row.best_repo_use}",
                f"- Merge directly into ACS/CPS: {mergeable}",
                f"- Why not: {row.reason_not_mergeable}",
                f"- Source: {row.primary_source_url}",
                "",
            ]
        )

    lines = [
        "# SCE Supplemental Expectations and Reservation-Wage Note",
        "",
        "## Why this module exists",
        "",
        "ACS, CPS ASEC, and ATUS can estimate realized earnings gaps, employment-selection sensitivity, and time-use channels, but they do not observe the wage workers say they would require before accepting a new job.",
        "The New York Fed Survey of Consumer Expectations (SCE) Labor Market Survey is the best free public dataset in this project scope for that narrower question because it directly elicits reservation wages, expected offer wages, offer receipt, and offer acceptance outcomes.",
        "",
        "## Bottom line for this repo",
        "",
        "- Use SCE as supporting evidence on bargaining thresholds, outside options, and expectations.",
        "- Do not treat SCE as a drop-in control variable inside the main ACS/CPS wage regressions.",
        "- The main reason is identification, not convenience: SCE is a separate survey with a different sampling frame, different timing, and expectation-based measures that cannot be person-level merged onto ACS/CPS/SIPP.",
        "",
        "## Official survey facts",
        "",
        f"- The SCE launched in 2013 as a New York Fed survey on expectations about inflation, labor markets, and household finance: {SCE_MAIN_URL}",
        f"- The Labor Market Survey module has been fielded since March 2014, first released in August 2017, and is fielded every four months in March, July, and November: {SCE_FAQ_URL}",
        f"- The Labor Market page states that the module collects experiences and expectations related to earnings, job transitions, and job offers, and offers chart data, a guide, questionnaire, and complete microdata downloads: {SCE_LABOR_URL}",
        f"- Recent New York Fed releases describe the labor module as surveying about 1,000 panelists each wave and explicitly tracking reservation wages, expected offers, and realized offers: {SCE_RESERVATION_WAGE_POST_URL}",
        "",
        "## Why SCE matters for the current findings",
        "",
        f"- The rebuilt ACS year-by-year `M5` gap averages {metrics['acs_m5_mean_pct_gap']:.2f}% and sits at {metrics['acs_2023_m5_pct_gap']:.2f}% in 2023. That is a realized worker-gap result, not a bargaining-threshold result.",
        f"- ACS selection robustness shows a much larger combined expected annual-earnings gap ({metrics['acs_s2_expected_gap_pct']:.2f}% mean in `S2`) than the IPW worker hourly gap ({metrics['acs_s2_ipw_gap_pct']:.2f}%).",
        f"- CPS selection shows the same pattern: combined expected annual-earnings gap {metrics['cps_s2_expected_gap_pct']:.2f}% versus IPW worker hourly gap {metrics['cps_s2_ipw_gap_pct']:.2f}%.",
        f"- ATUS adds mechanism evidence in the same direction, with women averaging {abs(metrics['atus_paid_work_gap_minutes']):.1f} fewer paid-work minutes per day, {metrics['atus_housework_gap_minutes']:.1f} more housework minutes, and {metrics['atus_childcare_gap_minutes']:.1f} more childcare minutes.",
        "",
        "Those results already show that realized earnings, hours, and labor-force attachment differ by sex. SCE complements them by getting closer to what workers expect, what offers they receive, and what minimum pay they say they require.",
        "",
        "## What SCE can and cannot identify",
        "",
        "### What it can add",
        "",
        "- Whether women report systematically lower reservation wages than men.",
        "- Whether women expect lower offer wages even before observed realized wage differences are measured.",
        "- Whether offer receipt, acceptance, or rejection patterns differ by sex.",
        "- Whether post-2020 changes in reservation wages track or diverge from realized wage-gap patterns.",
        "",
        "### What it cannot do cleanly",
        "",
        "- It cannot be merged person-by-person onto ACS, CPS ASEC, or ATUS.",
        "- It cannot retroactively become a control in the current ACS/CPS wage regressions.",
        "- It should not be used to claim that the adjusted gap disappears once we 'control for bargaining' unless the bargaining analysis is run inside SCE itself.",
        "",
        "## Recommended empirical use",
        "",
        "1. Keep ACS/CPS/ATUS as the main realized-gap evidence.",
        "2. Use SCE as a separate supplemental section on expectations, reservation wages, and job-offer dynamics.",
        "3. If SCE microdata are downloaded later, estimate sex differences in reservation wages and offer expectations within SCE itself, then compare their direction and magnitude to the ACS/CPS worker-gap results.",
        "4. Present any SCE findings as mechanism or calibration evidence, not as the final adjusted-gap estimand.",
        "",
        "## Measure map",
        "",
        *measure_lines,
        "## Download and staging",
        "",
        f"- The repo downloader writes acquisition instructions into `{DATA_RAW_DIR.relative_to(PROJECT_ROOT)}`.",
        f"- Databank / microdata access: {SCE_DATABANK_URL}",
        f"- Questionnaire: {SCE_QUESTIONNAIRE_URL}",
        f"- Codebook: {SCE_CODEBOOK_URL}",
        "",
        "## Practical interpretation",
        "",
        "If SCE shows lower female reservation wages or lower expected offer wages, that would support a bargaining/expectations channel. It still would not imply that the realized ACS/CPS worker gap is explained away; it would imply that part of the mechanism may run through expectations and outside options rather than only realized job sorting or hours.",
        "",
        "That is the defensible way to use SCE in this project.",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "diagnostics").mkdir(parents=True, exist_ok=True)

    # Ensure the staged acquisition guide exists in the canonical raw-data path.
    SCELaborMarketDownloader(raw_dir=DATA_RAW_DIR).download()

    measure_map = build_sce_measure_map()
    metrics = collect_repo_context()

    measure_map.to_csv(RESULTS_DIR / "diagnostics" / "sce_measure_map.csv", index=False)
    write_sce_report(metrics, measure_map, REPORTS_DIR / "sce_supplement.md")


if __name__ == "__main__":
    main()
