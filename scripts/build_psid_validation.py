#!/usr/bin/env python3
"""Build a PSID validation note from the processed 2023 main-panel extract."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from gender_gap.models.descriptive import gap_by_subgroup, raw_gap
from gender_gap.models.oaxaca import oaxaca_blinder, oaxaca_summary_table
from gender_gap.models.ols import design_source_columns, results_to_dataframe, run_sequential_ols
from gender_gap.standardize.psid_standardize import load_psid_2023


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
DIAG_DIR = PROJECT_ROOT / "results" / "diagnostics"


def _pct_from_log_coef(value: float) -> float:
    return abs((np.exp(float(value)) - 1.0) * 100.0)


def _categorical_term_column(term: str) -> str:
    match = re.match(r"C\((\w+)\)", term)
    return match.group(1) if match else term


def _available_controls(
    df: pd.DataFrame,
    controls: list[str],
    min_nonnull_share: float = 0.5,
) -> list[str]:
    usable: list[str] = []
    for term in controls:
        column = _categorical_term_column(term)
        if column not in df.columns:
            continue
        series = df[column]
        if series.dropna().empty:
            continue
        if series.notna().mean() < min_nonnull_share:
            continue
        if term.startswith("C("):
            if series.nunique(dropna=True) < 2:
                continue
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.dropna().nunique() < 2:
                continue
        usable.append(term)
    return usable


def _prepare_psid() -> pd.DataFrame:
    df = load_psid_2023().copy()
    df["age_sq"] = pd.to_numeric(df["age"], errors="coerce") ** 2
    df["log_hourly_wage_real"] = np.log(pd.to_numeric(df["hourly_wage_real"], errors="coerce"))
    df["ever_married"] = df["marital_status"].fillna("other").ne("never_married").astype(int)
    df["log_annual_earnings_real"] = np.log(
        pd.to_numeric(df["annual_earnings_real"], errors="coerce").where(
            pd.to_numeric(df["annual_earnings_real"], errors="coerce") > 0
        )
    )
    return df


def _winsorized_gap(
    df: pd.DataFrame,
    outcome: str,
    weight: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[dict[str, float], dict[str, float]]:
    sample = df.loc[pd.to_numeric(df[outcome], errors="coerce").gt(0)].copy()
    values = pd.to_numeric(sample[outcome], errors="coerce")
    lower = float(values.quantile(lower_q))
    upper = float(values.quantile(upper_q))
    sample[outcome] = values.clip(lower=lower, upper=upper)
    return raw_gap(sample, outcome=outcome, weight=weight), {
        "lower_bound": lower,
        "upper_bound": upper,
        "lower_quantile": lower_q,
        "upper_quantile": upper_q,
        "n": int(len(sample)),
    }


def _blocks_for_psid(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
    labels = {
        "P0": "female only",
        "P1": "age, race, and education",
        "P2": "state and job sorting",
        "P3": "work arrangement",
        "P4": "family structure",
        "P5": "reproductive stage",
    }
    blocks = {
        "P0": ["female"],
        "P1": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
        "P2": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)", "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)"],
        "P3": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)", "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)", "usual_hours_week", "work_from_home", "commute_minutes_one_way"],
        "P4": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)", "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)", "usual_hours_week", "work_from_home", "commute_minutes_one_way", "C(marital_status)", "ever_married", "number_children", "children_under_5"],
        "P5": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)", "C(state_fips)", "C(occupation_code)", "C(industry_code)", "C(class_of_worker)", "usual_hours_week", "work_from_home", "commute_minutes_one_way", "C(marital_status)", "ever_married", "number_children", "children_under_5", "recent_birth", "recent_marriage", "own_child_under6", "own_child_6_17_only", "C(couple_type)", "C(reproductive_stage)"],
    }
    return {k: _available_controls(df, v) for k, v in blocks.items()}, labels


def _run_models(df: pd.DataFrame, blocks: dict[str, list[str]]) -> pd.DataFrame:
    prepared = df.copy()
    final_model = list(blocks.values())[-1]
    required = ["log_hourly_wage_real", "person_weight"] + design_source_columns(final_model)
    mask = prepared["log_hourly_wage_real"].notna() & pd.to_numeric(prepared["person_weight"], errors="coerce").gt(0)
    for column in required:
        if column in {"log_hourly_wage_real", "person_weight"} or column not in prepared.columns:
            continue
        mask &= prepared[column].notna()
    prepared = prepared.loc[mask].copy()
    results = run_sequential_ols(
        prepared,
        outcome="log_hourly_wage_real",
        weight_col="person_weight",
        blocks=blocks,
    )
    summary = results_to_dataframe(results)
    summary["pct_gap"] = summary["female_coef"].apply(_pct_from_log_coef)
    summary["common_sample_n"] = len(prepared)
    return summary


def _transition_rows(summary: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    frame = summary.copy()
    frame["next_model"] = frame["model"].shift(-1)
    frame["next_pct_gap"] = frame["pct_gap"].shift(-1)
    frame["reduction_pp"] = frame["pct_gap"] - frame["next_pct_gap"]
    frame = frame.dropna(subset=["next_model", "next_pct_gap"]).copy()
    frame["added_block"] = frame["next_model"].map(labels)
    return frame[["model", "next_model", "added_block", "pct_gap", "next_pct_gap", "reduction_pp"]].rename(
        columns={"model": "from_model", "next_model": "to_model", "pct_gap": "gap_before_pct", "next_pct_gap": "gap_after_pct"}
    )


def _summary_rows(df: pd.DataFrame, summary: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    raw_hourly = raw_gap(df[df["hourly_wage_real"].gt(0)], outcome="hourly_wage_real", weight="person_weight")
    robust_hourly, robust_meta = _winsorized_gap(
        df,
        outcome="hourly_wage_real",
        weight="person_weight",
    )
    raw_annual = raw_gap(df[df["annual_earnings_real"].gt(0)], outcome="annual_earnings_real", weight="person_weight")
    biggest = transitions.sort_values("reduction_pp", ascending=False).iloc[0]
    common_sample_raw_gap = float(summary.loc[summary["model"] == "P0", "pct_gap"].iloc[0])
    final_gap = float(summary.loc[summary["model"] == summary["model"].iloc[-1], "pct_gap"].iloc[0])
    rows = [
        {"metric": "descriptive_hourly_gap_pct", "value": float(raw_hourly["gap_pct"])},
        {"metric": "winsorized_hourly_gap_pct", "value": float(robust_hourly["gap_pct"])},
        {"metric": "winsorized_hourly_lower_bound", "value": float(robust_meta["lower_bound"])},
        {"metric": "winsorized_hourly_upper_bound", "value": float(robust_meta["upper_bound"])},
        {"metric": "raw_annual_gap_pct", "value": float(raw_annual["gap_pct"])},
        {"metric": "common_sample_raw_hourly_gap_pct", "value": common_sample_raw_gap},
        {"metric": "final_hourly_gap_pct", "value": final_gap},
        {"metric": "largest_reduction_block", "value": str(biggest["added_block"])},
        {"metric": "largest_reduction_pp", "value": float(biggest["reduction_pp"])},
        {"metric": "common_sample_n", "value": int(summary["common_sample_n"].iloc[0])},
        {"metric": "psid_rows_total", "value": int(len(df))},
    ]
    for row in transitions.itertuples(index=False):
        rows.append({"metric": f"{row.to_model}_reduction_pp", "value": float(row.reduction_pp)})
    return pd.DataFrame(rows)


def build_report(
    summary_metrics: pd.DataFrame,
    transitions: pd.DataFrame,
    oaxaca_summary: pd.DataFrame,
    stage_gap: pd.DataFrame,
) -> str:
    metrics = summary_metrics.set_index("metric")["value"]
    top_stage = stage_gap.sort_values("gap_pct", ascending=False).head(3)
    top_blocks = transitions.sort_values("reduction_pp", ascending=False).head(2)
    lines = [
        "# PSID Validation",
        "",
        "This note uses the new 2023 PSID main-panel validation extract. It is a one-year cross-section built from the public family and cross-year individual files, limited to reference persons and spouses.",
        "",
        "Important scope limits:",
        "- this is not yet a multi-wave PSID panel analysis",
        "- the file covers reference persons and spouses rather than the full household roster",
        "",
        f"- Descriptive hourly-wage gap: {float(metrics['descriptive_hourly_gap_pct']):.2f}%",
        f"- Winsorized hourly-wage gap (p1-p99): {float(metrics['winsorized_hourly_gap_pct']):.2f}%",
        f"- Raw annual-earnings gap: {float(metrics['raw_annual_gap_pct']):.2f}%",
        f"- Common-sample model raw hourly gap: {float(metrics['common_sample_raw_hourly_gap_pct']):.2f}%",
        f"- Final staged hourly gap: {float(metrics['final_hourly_gap_pct']):.2f}%",
        f"- Common complete-case sample: {int(metrics['common_sample_n'])}",
        f"- Largest reduction block: {metrics['largest_reduction_block']} ({float(metrics['largest_reduction_pp']):.2f} percentage points)",
        "",
        "## Block Transitions",
        "",
    ]
    for row in top_blocks.itertuples(index=False):
        lines.append(
            f"- {row.added_block}: {row.gap_before_pct:.2f}% -> {row.gap_after_pct:.2f}% ({row.reduction_pp:.2f} points)"
        )
    lines.extend(["", "## Oaxaca Snapshot", ""])
    for row in oaxaca_summary.itertuples(index=False):
        lines.append(f"- {row.component}: {row.pct:.2f}%")
    lines.extend(["", "## Reproductive-Stage Gaps", ""])
    for row in top_stage.itertuples(index=False):
        lines.append(
            f"- {row.group}: {row.gap_pct:.2f}% ({row.male_mean:.2f} vs {row.female_mean:.2f})"
        )
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "PSID now provides a live validation lane for the reproductive-burden extension. In its current form it is best read as a compact cross-check on the ACS story: job sorting, work arrangement, and family/reproductive variables all matter, but the lane is still narrower than ACS because its public-use demographic surface is coarser and it is not yet a full multi-wave panel treatment.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    df = _prepare_psid()
    blocks, labels = _blocks_for_psid(df)
    summary = _run_models(df, blocks)
    transitions = _transition_rows(summary, labels)

    oaxaca_controls = [
        "age",
        "age_sq",
        "usual_hours_week",
        "work_from_home",
        "commute_minutes_one_way",
        "number_children",
        "children_under_5",
        "recent_birth",
    ]
    oaxaca_df = df.loc[
        df["log_hourly_wage_real"].notna() & pd.to_numeric(df["person_weight"], errors="coerce").gt(0),
        ["female", "log_hourly_wage_real", "person_weight"] + [c for c in oaxaca_controls if c in df.columns],
    ].copy()
    ob = oaxaca_blinder(
        oaxaca_df,
        outcome="log_hourly_wage_real",
        controls=[c for c in oaxaca_controls if c in oaxaca_df.columns],
        weight_col="person_weight",
    )
    oaxaca_summary = oaxaca_summary_table(ob)

    stage_gap = gap_by_subgroup(
        df[df["hourly_wage_real"].gt(0)].copy(),
        group_col="reproductive_stage",
        outcome="hourly_wage_real",
        weight="person_weight",
    ).sort_values("gap_pct", ascending=False)

    summary_metrics = _summary_rows(df, summary, transitions)
    summary_metrics.loc[:, "metric_group"] = np.where(
        summary_metrics["metric"].str.startswith("winsorized_hourly_"),
        "robust_hourly_gap",
        "core",
    )

    summary.to_csv(DIAG_DIR / "psid_ols_validation.csv", index=False)
    transitions.to_csv(DIAG_DIR / "psid_factor_contributions.csv", index=False)
    summary_metrics.to_csv(DIAG_DIR / "psid_validation_summary.csv", index=False)
    oaxaca_summary.to_csv(DIAG_DIR / "psid_oaxaca_summary.csv", index=False)
    stage_gap.to_csv(DIAG_DIR / "psid_reproductive_stage_gaps.csv", index=False)

    report = build_report(summary_metrics, transitions, oaxaca_summary, stage_gap)
    (REPORTS_DIR / "psid_validation.md").write_text(report + "\n", encoding="utf-8")

    print(f"Wrote {DIAG_DIR / 'psid_ols_validation.csv'}")
    print(f"Wrote {DIAG_DIR / 'psid_factor_contributions.csv'}")
    print(f"Wrote {DIAG_DIR / 'psid_validation_summary.csv'}")
    print(f"Wrote {DIAG_DIR / 'psid_oaxaca_summary.csv'}")
    print(f"Wrote {DIAG_DIR / 'psid_reproductive_stage_gaps.csv'}")
    print(f"Wrote {REPORTS_DIR / 'psid_validation.md'}")


if __name__ == "__main__":
    main()
