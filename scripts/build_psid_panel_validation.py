#!/usr/bin/env python3
"""Build a compact by-year PSID validation note from the stacked panel extract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gender_gap.models.descriptive import raw_gap
from gender_gap.models.ols import design_source_columns, results_to_dataframe, run_sequential_ols
from gender_gap.standardize.psid_standardize import build_psid_panel_analysis_file, load_psid_panel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
DIAG_DIR = PROJECT_ROOT / "results" / "diagnostics"


def _pct_from_log_coef(value: float) -> float:
    return abs((np.exp(float(value)) - 1.0) * 100.0)


def _blocks() -> dict[str, list[str]]:
    return {
        "P0": ["female"],
        "P1": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
        "P2": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)", "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)"],
        "P3": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)", "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)", "usual_hours_week", "work_from_home", "commute_minutes_one_way"],
        "P4": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)", "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)", "usual_hours_week", "work_from_home", "commute_minutes_one_way", "C(marital_status)", "number_children", "children_under_5"],
        "P5": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)", "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)", "usual_hours_week", "work_from_home", "commute_minutes_one_way", "C(marital_status)", "number_children", "children_under_5", "recent_birth", "recent_marriage", "own_child_under6", "own_child_6_17_only", "C(couple_type)", "C(reproductive_stage)"],
    }


def _prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_sq"] = pd.to_numeric(out["age"], errors="coerce") ** 2
    out["log_hourly_wage_real"] = np.log(pd.to_numeric(out["hourly_wage_real"], errors="coerce"))
    return out


def _winsorized_hourly_gap(
    df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[dict[str, float], dict[str, float]]:
    sample = df.loc[pd.to_numeric(df["hourly_wage_real"], errors="coerce").gt(0)].copy()
    values = pd.to_numeric(sample["hourly_wage_real"], errors="coerce")
    lower = float(values.quantile(lower_q))
    upper = float(values.quantile(upper_q))
    sample["hourly_wage_real"] = values.clip(lower=lower, upper=upper)
    return raw_gap(sample, outcome="hourly_wage_real", weight="person_weight"), {
        "lower_bound": lower,
        "upper_bound": upper,
    }


def _run_year_models(df: pd.DataFrame) -> pd.DataFrame:
    blocks = _blocks()
    final_controls = design_source_columns(blocks["P5"])
    mask = df["log_hourly_wage_real"].notna() & pd.to_numeric(df["person_weight"], errors="coerce").gt(0)
    for column in final_controls:
        if column in df.columns:
            mask &= df[column].notna()
    sample = df.loc[mask].copy()
    summary = results_to_dataframe(
        run_sequential_ols(sample, outcome="log_hourly_wage_real", weight_col="person_weight", blocks=blocks)
    )
    summary["pct_gap"] = summary["female_coef"].apply(_pct_from_log_coef)
    summary["common_sample_n"] = len(sample)
    return summary


def build_report(trend: pd.DataFrame) -> str:
    indexed = trend.set_index("survey_year")
    years = sorted(indexed.index.tolist())
    lines = [
        "# PSID Panel Validation",
        "",
        "This note extends the PSID validation lane from a single 2023 cross-section to a stacked 2021/2023 file built from the main public family and cross-year individual records.",
        "",
    ]
    for year in years:
        row = indexed.loc[year]
        lines.extend(
            [
                f"## {year}",
                "",
                f"- Descriptive hourly-wage gap: {row['descriptive_hourly_gap_pct']:.2f}%",
                f"- Winsorized hourly-wage gap (p1-p99): {row['winsorized_hourly_gap_pct']:.2f}%",
                f"- Raw annual-earnings gap: {row['raw_annual_gap_pct']:.2f}%",
                f"- Common-sample raw hourly gap: {row['common_sample_raw_hourly_gap_pct']:.2f}%",
                f"- Final staged hourly gap: {row['final_hourly_gap_pct']:.2f}%",
                f"- Largest reduction block: {row['largest_reduction_block']} ({row['largest_reduction_pp']:.2f} percentage points)",
                f"- Common sample size: {int(row['common_sample_n'])}",
                "",
            ]
        )
    if len(years) >= 2:
        first, last = years[0], years[-1]
        lines.extend(
            [
                "## Change Over Time",
                "",
                f"- Descriptive hourly gap changed from {indexed.loc[first, 'descriptive_hourly_gap_pct']:.2f}% in {first} to {indexed.loc[last, 'descriptive_hourly_gap_pct']:.2f}% in {last}.",
                f"- Winsorized hourly gap changed from {indexed.loc[first, 'winsorized_hourly_gap_pct']:.2f}% in {first} to {indexed.loc[last, 'winsorized_hourly_gap_pct']:.2f}% in {last}.",
                f"- Final staged hourly gap changed from {indexed.loc[first, 'final_hourly_gap_pct']:.2f}% in {first} to {indexed.loc[last, 'final_hourly_gap_pct']:.2f}% in {last}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Takeaway",
            "",
            "The stacked PSID panel keeps the same scope limits as the one-year validation lane, but it is enough to check whether the broad paygap patterns are stable across adjacent public waves rather than being a one-off 2023 artifact.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    panel_path = PROJECT_ROOT / "data" / "external" / "psid" / "psid_2021_2023_analysis_ready.parquet"
    if not panel_path.exists():
        build_psid_panel_analysis_file()
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        shared_path = PROJECT_ROOT.parent / "data" / "sources" / "umich" / "psid" / "main_public" / "paygap" / "processed" / "psid" / "psid_2021_2023_analysis_ready.parquet"
        if panel_path.exists() or panel_path.is_symlink():
            panel_path.unlink()
        panel_path.symlink_to(shared_path)
    df = _prepare_panel(load_psid_panel(panel_path))

    rows: list[dict[str, object]] = []
    labels = {
        "P1": "age, race, and education",
        "P2": "state and job sorting",
        "P3": "work arrangement",
        "P4": "family structure",
        "P5": "reproductive stage",
    }
    for year, year_df in df.groupby("survey_year", observed=True):
        desc = raw_gap(year_df[year_df["hourly_wage_real"].gt(0)], outcome="hourly_wage_real", weight="person_weight")
        robust_desc, robust_meta = _winsorized_hourly_gap(year_df)
        annual = raw_gap(year_df[year_df["annual_earnings_real"].gt(0)], outcome="annual_earnings_real", weight="person_weight")
        summary = _run_year_models(year_df)
        transition = summary[["model", "pct_gap"]].copy()
        transition["to_model"] = transition["model"].shift(-1)
        transition["next_gap"] = transition["pct_gap"].shift(-1)
        transition["reduction_pp"] = transition["pct_gap"] - transition["next_gap"]
        transition = transition.dropna(subset=["to_model", "reduction_pp"]).sort_values("reduction_pp", ascending=False)
        largest = transition.iloc[0]
        rows.append(
            {
                "survey_year": int(year),
                "descriptive_hourly_gap_pct": float(desc["gap_pct"]),
                "winsorized_hourly_gap_pct": float(robust_desc["gap_pct"]),
                "winsorized_hourly_lower_bound": float(robust_meta["lower_bound"]),
                "winsorized_hourly_upper_bound": float(robust_meta["upper_bound"]),
                "raw_annual_gap_pct": float(annual["gap_pct"]),
                "common_sample_raw_hourly_gap_pct": float(summary.loc[summary["model"] == "P0", "pct_gap"].iloc[0]),
                "final_hourly_gap_pct": float(summary.loc[summary["model"] == "P5", "pct_gap"].iloc[0]),
                "largest_reduction_block": labels[str(largest["to_model"])],
                "largest_reduction_pp": float(largest["reduction_pp"]),
                "common_sample_n": int(summary["common_sample_n"].iloc[0]),
            }
        )
        summary.to_csv(DIAG_DIR / f"psid_{int(year)}_ols_validation.csv", index=False)

    trend = pd.DataFrame(rows).sort_values("survey_year")
    trend.to_csv(DIAG_DIR / "psid_panel_trend_summary.csv", index=False)
    report = build_report(trend)
    (REPORTS_DIR / "psid_panel_validation.md").write_text(report + "\n", encoding="utf-8")

    print(f"Wrote {DIAG_DIR / 'psid_panel_trend_summary.csv'}")
    print(f"Wrote {REPORTS_DIR / 'psid_panel_validation.md'}")


if __name__ == "__main__":
    main()
