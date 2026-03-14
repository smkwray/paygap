#!/usr/bin/env python3
"""Build a report guide with dataset-specific interpretation notes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from gender_gap.models.oaxaca import oaxaca_blinder
from gender_gap.standardize.nlsy_standardize import (
    load_nlsy79,
    load_nlsy97,
    standardize_nlsy79_for_gap,
    standardize_nlsy97_for_gap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DIAG_DIR = RESULTS_DIR / "diagnostics"
NLSY_DIR = Path(os.environ.get("NLSY_DATA_DIR", "data/external/nlsy"))


def _pct_from_log_coef(value: float) -> float:
    return abs((np.exp(float(value)) - 1.0) * 100.0)


def _largest_gap_shift(ols: pd.DataFrame, labels: dict[str, str]) -> tuple[str, float]:
    frame = ols.copy()
    if "pct_gap" not in frame.columns:
        frame["pct_gap"] = frame["female_coef"].apply(_pct_from_log_coef)
    frame = frame[["model", "pct_gap"]].copy()
    frame["next_model"] = frame["model"].shift(-1)
    frame["next_pct_gap"] = frame["pct_gap"].shift(-1)
    frame["reduction_pp"] = frame["pct_gap"] - frame["next_pct_gap"]
    frame = frame.dropna(subset=["next_model", "next_pct_gap"])
    best = frame.sort_values("reduction_pp", ascending=False).iloc[0]
    return labels.get(str(best["next_model"]), str(best["next_model"])), float(best["reduction_pp"])


def _simple_oaxaca_contributors(
    path: Path,
    controls: list[str],
    extra_filter=None,
    extra_columns: list[str] | None = None,
) -> list[tuple[str, float]]:
    cols = ["female", "person_weight", "hourly_wage_real"] + controls
    if extra_columns:
        cols.extend(extra_columns)
    cols = list(dict.fromkeys(cols))
    df = pd.read_parquet(path, columns=cols)
    df["log_hourly_wage_real"] = np.log(df["hourly_wage_real"].replace(0, np.nan))
    mask = df["hourly_wage_real"].gt(0) & df["log_hourly_wage_real"].notna()
    if extra_filter is not None:
        mask &= extra_filter(df)
    df = df.loc[mask].copy()
    result = oaxaca_blinder(df, outcome="log_hourly_wage_real", controls=controls, weight_col="person_weight")
    contrib = result.contributions.copy()
    contrib = contrib.loc[contrib["variable"] != "const"].copy()
    contrib["abs_contribution"] = contrib["contribution"].abs()
    top = contrib.sort_values("abs_contribution", ascending=False).head(3)
    return [(str(row.variable), float(row.contribution_pct)) for row in top.itertuples(index=False)]


def _nlsy_summary() -> pd.DataFrame:
    deep_dive_path = DIAG_DIR / "nlsy_cohort_comparison.csv"
    if deep_dive_path.exists():
        wide = pd.read_csv(deep_dive_path)
        rows = []
        for row in wide.itertuples(index=False):
            for metric in [
                "raw_gap_pct",
                "skills_reduction_pp",
                "background_reduction_pp",
                "occupation_reduction_pp",
                "family_reduction_pp",
                "resources_reduction_pp",
                "largest_reduction_pp",
                "final_gap_pct",
            ]:
                rows.append({"dataset": row.dataset, "metric": metric, "value": float(getattr(row, metric))})
            rows.append(
                {
                    "dataset": row.dataset,
                    "metric": "largest_reduction_block",
                    "value": str(row.largest_reduction_block),
                }
            )
        return pd.DataFrame(rows)

    rows = []
    for label, loader, standardizer in [
        ("NLSY79", load_nlsy79, standardize_nlsy79_for_gap),
        ("NLSY97", load_nlsy97, standardize_nlsy97_for_gap),
    ]:
        raw = loader(NLSY_DIR / f"{label.lower()}_cfa_resid.csv")
        df = standardizer(raw)
        df = df[df["annual_earnings_real"].notna() & (df["annual_earnings_real"] > 0)].copy()
        df["hourly_wage_real"] = df["annual_earnings_real"]
        df["log_hourly_wage_real"] = np.log(df["hourly_wage_real"])

        from gender_gap.models.ols import NLSY_BLOCK_DEFINITIONS, results_to_dataframe, run_sequential_ols

        ols = results_to_dataframe(
            run_sequential_ols(
                df,
                outcome="log_hourly_wage_real",
                weight_col="person_weight",
                blocks=NLSY_BLOCK_DEFINITIONS,
            )
        )
        ols["pct_gap"] = ols["female_coef"].apply(_pct_from_log_coef)
        get = lambda model: float(ols.loc[ols["model"] == model, "pct_gap"].iloc[0])
        rows.extend(
            [
                {"dataset": label, "metric": "N1_pct_gap", "value": get("N1")},
                {"dataset": label, "metric": "N2_pct_gap", "value": get("N2")},
                {"dataset": label, "metric": "g_proxy_reduction_pp", "value": get("N1") - get("N2")},
                {"dataset": label, "metric": "occupation_reduction_pp", "value": get("N2") - get("N3")},
                {"dataset": label, "metric": "N5_pct_gap", "value": get("N5")},
            ]
        )
    return pd.DataFrame(rows)


def build_dataset_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acs_ols = pd.read_csv(RESULTS_DIR / "acs" / "2023" / "ols_sequential.csv")
    cps_ols = pd.read_csv(RESULTS_DIR / "cps" / "2023" / "ols_sequential.csv")
    sipp_ols = pd.read_csv(RESULTS_DIR / "sipp" / "2023" / "ols_sequential.csv")
    acs_raw = pd.read_csv(RESULTS_DIR / "acs" / "2023" / "raw_gap.csv").iloc[0]
    cps_raw = pd.read_csv(RESULTS_DIR / "cps" / "2023" / "raw_gap.csv").iloc[0]
    sipp_raw = pd.read_csv(RESULTS_DIR / "sipp" / "2023" / "raw_gap.csv").iloc[0]

    acs_shift = _largest_gap_shift(
        acs_ols,
        {"M1": "demographics", "M2": "geography", "M3": "job sorting", "M4": "schedule/commute", "M5": "family"},
    )
    cps_shift = _largest_gap_shift(
        cps_ols,
        {"M1": "demographics", "M2": "geography", "M3": "job sorting", "M_full": "family"},
    )
    sipp_shift = _largest_gap_shift(
        sipp_ols,
        {"SIPP1": "month effects", "SIPP2": "job sorting", "SIPP3": "hours/job structure"},
    )

    acs_top = _simple_oaxaca_contributors(
        PROJECT_ROOT / "data" / "processed" / "acs_2023_analysis_ready.parquet",
        ["age", "age_sq", "usual_hours_week", "work_from_home", "commute_minutes_one_way", "number_children", "children_under_5"],
    )
    cps_top = _simple_oaxaca_contributors(
        PROJECT_ROOT / "data" / "processed" / "cps_asec_2023_analysis_ready.parquet",
        ["age", "age_sq", "usual_hours_week", "number_children", "children_under_5"],
    )
    sipp_top = _simple_oaxaca_contributors(
        PROJECT_ROOT / "data" / "processed" / "sipp_standardized.parquet",
        ["usual_hours_week", "actual_hours_last_week", "paid_hourly", "multiple_jobholder"],
        extra_filter=lambda df: df["employed"].fillna(0).eq(1),
        extra_columns=["employed"],
    )

    dataset_rows = [
        {
            "dataset": "ACS",
            "year": 2023,
            "headline_gap_pct": float(acs_raw["gap_pct"]),
            "adjusted_gap_pct": _pct_from_log_coef(float(acs_ols.loc[acs_ols["model"] == "M5", "female_coef"].iloc[0])),
            "main_explaining_block": acs_shift[0],
            "block_reduction_pp": acs_shift[1],
            "top_oaxaca_contributors": "; ".join(f"{name} ({pct:.2f}%)" for name, pct in acs_top),
            "missing_key_factors": "firm/pay-setting, tenure, exact employer, detailed within-job task mix, direct bargaining measures",
        },
        {
            "dataset": "CPS ASEC",
            "year": 2023,
            "headline_gap_pct": float(cps_raw["gap_pct"]),
            "adjusted_gap_pct": _pct_from_log_coef(float(cps_ols.loc[cps_ols["model"] == "M_full", "female_coef"].iloc[0])),
            "main_explaining_block": cps_shift[0],
            "block_reduction_pp": cps_shift[1],
            "top_oaxaca_contributors": "; ".join(f"{name} ({pct:.2f}%)" for name, pct in cps_top),
            "missing_key_factors": "commute, firm effects, direct bargaining measures, richer job-quality variables",
        },
        {
            "dataset": "SIPP",
            "year": 2023,
            "headline_gap_pct": float(sipp_raw["gap_pct"]),
            "adjusted_gap_pct": float(abs(sipp_ols.loc[sipp_ols["model"] == "SIPP3", "pct_gap"].iloc[0])),
            "main_explaining_block": sipp_shift[0],
            "block_reduction_pp": sipp_shift[1],
            "top_oaxaca_contributors": "; ".join(f"{name} ({pct:.2f}%)" for name, pct in sipp_top),
            "missing_key_factors": "state geography in current release, rich family controls, commute, firm/pay-setting variables",
        },
        {
            "dataset": "ATUS",
            "year": 2023,
            "headline_gap_pct": np.nan,
            "adjusted_gap_pct": np.nan,
            "main_explaining_block": "time allocation",
            "block_reduction_pp": np.nan,
            "top_oaxaca_contributors": "paid work (-67.96 min/day); housework (+31.91); childcare (+11.37)",
            "missing_key_factors": "no direct wage equation, no person-level merge to ACS/CPS/SIPP",
        },
        {
            "dataset": "SCE",
            "year": 2025,
            "headline_gap_pct": np.nan,
            "adjusted_gap_pct": np.nan,
            "main_explaining_block": "expectations/outside options",
            "block_reduction_pp": np.nan,
            "top_oaxaca_contributors": "expected offer gap (+17.46); reservation wage gap (+21.34)",
            "missing_key_factors": "cannot merge to main wage files; smaller expectation-based survey",
        },
    ]
    return pd.DataFrame(dataset_rows), _nlsy_summary(), pd.DataFrame(
        [
            {"priority": 1, "variable_family": "firm and employer effects", "why": "likely large residual source within occupation/industry"},
            {"priority": 2, "variable_family": "tenure and work-history interruptions", "why": "important for promotion and wage progression"},
            {"priority": 3, "variable_family": "direct bargaining / reservation wages", "why": "closest to ask-wage channel; SCE helps here"},
            {"priority": 4, "variable_family": "schedule predictability / flexibility", "why": "can change job sorting and wage penalties"},
            {"priority": 5, "variable_family": "within-occupation task specialization", "why": "broad job titles can hide meaningful role differences"},
        ]
    )


def build_report(dataset_summary: pd.DataFrame, nlsy_summary: pd.DataFrame, future_vars: pd.DataFrame) -> str:
    ds = dataset_summary.set_index("dataset")
    n79 = nlsy_summary[nlsy_summary["dataset"] == "NLSY79"].set_index("metric")["value"]
    n97 = nlsy_summary[nlsy_summary["dataset"] == "NLSY97"].set_index("metric")["value"]

    lines = [
        "# Reports Guide",
        "",
        "This guide explains what each dataset contributes, which observed factors matter most in the current data, what is missing, and where the repo can expand next.",
        "",
        "## How to read the current results",
        "",
        "- `Sequential OLS` tells you which blocks of observed variables reduce the residual gap the most.",
        "- `Oaxaca-style contributions` tell you which measured characteristics account for the explained part of the gap in a simple decomposition.",
        "- `Mechanism datasets` like ATUS and SCE do not replace the main wage regressions; they help interpret them.",
        "",
        "## ACS",
        "",
        "- Role: primary headline dataset for repeated-year worker-gap estimates.",
        f"- 2023 headline results: raw gap {ds.loc['ACS', 'headline_gap_pct']:.2f}%, adjusted gap {ds.loc['ACS', 'adjusted_gap_pct']:.2f}%.",
        "- Key wage-gap variables in use: age, race/ethnicity, education, state, occupation, industry, class of worker, hours, work from home, commute time, marital status, children.",
        f"- Largest observed gap reduction comes from: {ds.loc['ACS', 'main_explaining_block']} ({ds.loc['ACS', 'block_reduction_pp']:.2f} percentage points).",
        f"- Top simple explained-gap contributors: {ds.loc['ACS', 'top_oaxaca_contributors']}.",
        f"- What is missing: {ds.loc['ACS', 'missing_key_factors']}.",
        "- Main reports: `reports/findings_summary.md`, `reports/defensibility_report.md`, `reports/oaxaca_review.md`.",
        "",
        "## CPS ASEC",
        "",
        "- Role: external cross-check on the realized worker gap and selection robustness.",
        f"- 2023 headline results: raw gap {ds.loc['CPS ASEC', 'headline_gap_pct']:.2f}%, adjusted gap {ds.loc['CPS ASEC', 'adjusted_gap_pct']:.2f}%.",
        "- Key wage-gap variables in use: age, race/ethnicity, education, state, occupation, industry, class of worker, family variables.",
        f"- Largest observed gap reduction comes from: {ds.loc['CPS ASEC', 'main_explaining_block']} ({ds.loc['CPS ASEC', 'block_reduction_pp']:.2f} percentage points).",
        f"- Top simple explained-gap contributors: {ds.loc['CPS ASEC', 'top_oaxaca_contributors']}.",
        f"- What is missing: {ds.loc['CPS ASEC', 'missing_key_factors']}.",
        "- Main reports: `reports/selection_robustness.md`, `reports/final_synthesis.md`.",
        "",
        "## SIPP",
        "",
        "- Role: validated public-use extension dataset with a first descriptive and adjusted-gap surface.",
        f"- 2023 headline results: raw gap {ds.loc['SIPP', 'headline_gap_pct']:.2f}%, adjusted gap {ds.loc['SIPP', 'adjusted_gap_pct']:.2f}%.",
        "- Key wage-gap variables in use: month, occupation, industry, hours, paid-hourly status, multiple-jobholder status.",
        f"- Largest observed gap reduction comes from: {ds.loc['SIPP', 'main_explaining_block']} ({ds.loc['SIPP', 'block_reduction_pp']:.2f} percentage points).",
        f"- Top simple explained-gap contributors: {ds.loc['SIPP', 'top_oaxaca_contributors']}.",
        f"- What is missing: {ds.loc['SIPP', 'missing_key_factors']}.",
        "- Main reports: `reports/sipp_validation.md`, `reports/sipp_snapshot.md`, `reports/sipp_models.md`.",
        "",
        "## ATUS",
        "",
        "- Role: mechanism evidence on time allocation, not a wage regression.",
        f"- Key observed channels: {ds.loc['ATUS', 'top_oaxaca_contributors']}.",
        f"- What is missing: {ds.loc['ATUS', 'missing_key_factors']}.",
        "- Main report: `reports/m6_time_use_bridge.md`.",
        "",
        "## SCE",
        "",
        "- Role: mechanism evidence on expectations, reservation wages, and outside options.",
        f"- Key observed channels: {ds.loc['SCE', 'top_oaxaca_contributors']}.",
        "- Subgroup read: the gender gaps are positive in every public wave, but the education and income spreads are even larger than the gender spread.",
        f"- What is missing: {ds.loc['SCE', 'missing_key_factors']}.",
        "- Main reports: `reports/sce_supplement.md`, `reports/sce_public_analysis.md`, `reports/sce_subgroup_analysis.md`.",
        "",
        "## NLSY Sub-Analysis",
        "",
        "- Role: separate cohort lane with richer background and life-course variables than the cross-sectional public files, not just `g_proxy`.",
        f"- NLSY79: the biggest measured reductions come from {n79['largest_reduction_block']} ({float(n79['largest_reduction_pp']):.2f} points) and the skills/traits block ({float(n79['skills_reduction_pp']):.2f} points); final deep-model gap is {float(n79['final_gap_pct']):.2f}%.",
        f"- NLSY97: the biggest measured reductions come from {n97['largest_reduction_block']} ({float(n97['largest_reduction_pp']):.2f} points) and the skills/achievement block ({float(n97['skills_reduction_pp']):.2f} points); final deep-model gap is {float(n97['final_gap_pct']):.2f}%.",
        "- Caution: the NLSY97 adult-resources block is later-life and mechanism-sensitive, so it should not be read as a clean pre-market explanation.",
        f"- Family-background reductions are small in both cohorts ({float(n79['background_reduction_pp']):.2f} and {float(n97['background_reduction_pp']):.2f} points).",
        "- Interpretation: NLSY is most useful as a richer cohort/background check. Skills matter, but occupation and later-life structure still matter too, and the gap remains after all blocks are added.",
        "- Main report: `reports/nlsy_deep_dive.md`.",
        "",
        "## Future Variables To Test",
        "",
    ]
    for row in future_vars.itertuples(index=False):
        lines.append(f"- {int(row.priority)}. {row.variable_family}: {row.why}.")
    lines.extend([
        "",
        "## Bottom line",
        "",
        "Yes, the repo can already extract a useful answer to 'what in the data explains part of the gap?'. In the current files, occupation/industry job sorting is the biggest measured reduction in ACS and CPS ASEC 2023, while hours and job-structure variables are the biggest measured reduction in SIPP. Commute time matters where observed, and the biggest remaining blind spots are employer effects, tenure/interruption histories, direct bargaining measures, and finer within-job differences.",
    ])
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    dataset_summary, nlsy_summary, future_vars = build_dataset_summary()
    dataset_summary.to_csv(DIAG_DIR / "report_dataset_summary.csv", index=False)
    nlsy_summary.to_csv(DIAG_DIR / "nlsy_subanalysis_summary.csv", index=False)
    future_vars.to_csv(DIAG_DIR / "future_variables_to_test.csv", index=False)

    readme = build_report(dataset_summary, nlsy_summary, future_vars)
    (REPORTS_DIR / "README.md").write_text(readme + "\n", encoding="utf-8")
    print(f"Wrote {DIAG_DIR / 'report_dataset_summary.csv'}")
    print(f"Wrote {DIAG_DIR / 'nlsy_subanalysis_summary.csv'}")
    print(f"Wrote {DIAG_DIR / 'future_variables_to_test.csv'}")
    print(f"Wrote {REPORTS_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
