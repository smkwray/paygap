#!/usr/bin/env python3
"""Build a richer NLSY cohort analysis beyond the g_proxy-only summary."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from gender_gap.models.ols import (
    design_source_columns,
    results_to_dataframe,
    run_sequential_ols,
)
from gender_gap.standardize.nlsy_standardize import (
    load_nlsy79,
    load_nlsy97,
    standardize_nlsy79_for_gap,
    standardize_nlsy97_for_gap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
DIAG_DIR = PROJECT_ROOT / "results" / "diagnostics"

NEGATIVE_SENTINELS = {-1, -2, -3, -4, -5}
SAT_ACT_COLS = ["sat_math_bin", "sat_verbal_bin", "act_bin"]


def _pct_from_log_coef(value: float) -> float:
    return abs((np.exp(float(value)) - 1.0) * 100.0)


def _clean_sentinel(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(numeric.isin(NEGATIVE_SENTINELS))


def _positive_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.log(numeric.where(numeric > 0))


def _signed_log1p(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.sign(numeric) * np.log1p(np.abs(numeric))


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


def _make_transition_rows(
    dataset: str,
    cohort_label: str,
    summary: pd.DataFrame,
    model_labels: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frame = summary.copy()
    frame["pct_gap"] = frame["female_coef"].apply(_pct_from_log_coef)
    frame["next_model"] = frame["model"].shift(-1)
    frame["next_pct_gap"] = frame["pct_gap"].shift(-1)
    frame["reduction_pp"] = frame["pct_gap"] - frame["next_pct_gap"]
    frame = frame.dropna(subset=["next_model", "next_pct_gap"])
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "dataset": dataset,
                "cohort_label": cohort_label,
                "from_model": row.model,
                "to_model": row.next_model,
                "added_block": model_labels[str(row.next_model)],
                "gap_before_pct": float(row.pct_gap),
                "gap_after_pct": float(row.next_pct_gap),
                "reduction_pp": float(row.reduction_pp),
                "n_obs_after": int(
                    summary.loc[summary["model"] == row.next_model, "n_obs"].iloc[0]
                ),
            }
        )
    return rows


def _prepare_nlsy79() -> pd.DataFrame:
    raw = load_nlsy79()
    std = standardize_nlsy79_for_gap(raw)
    raw = raw.copy()
    raw["person_id"] = raw["person_id"].astype(str)
    extra = pd.DataFrame(
        {
            "person_id": raw["person_id"],
            "mother_education": _clean_sentinel(raw["mother_education"]),
            "father_education": _clean_sentinel(raw["father_education"]),
            "rotter_locus_control": _clean_sentinel(raw["rotter_locus_control_1979"]),
            "self_esteem": _clean_sentinel(raw["rosenberg_self_esteem_1980"]),
            "homeowner": _clean_sentinel(raw["homeowner_2000"]),
            "age_first_birth": _clean_sentinel(raw["age_first_birth_2000"]),
            "age_first_marriage": _clean_sentinel(raw["age_first_marriage_2000"]),
        }
    )
    df = std.merge(extra, on="person_id", how="left")
    df["ever_married"] = df["marital_status"].fillna("unknown").ne("never_married").astype(int)
    df["log_household_income"] = _positive_log(df["household_income"])
    df["signed_log_net_worth"] = _signed_log1p(df["net_worth"])
    df["log_annual_earnings_real"] = _positive_log(df["annual_earnings_real"])
    return df


def _prepare_nlsy97() -> pd.DataFrame:
    raw = load_nlsy97()
    std = standardize_nlsy97_for_gap(raw)
    raw = raw.copy()
    raw["person_id"] = raw["person_id"].astype(str)
    extra = pd.DataFrame(
        {
            "person_id": raw["person_id"],
            "mother_education": _clean_sentinel(raw["mother_education"]),
            "father_education": _clean_sentinel(raw["father_education"]),
            "num_marriages": pd.to_numeric(raw["num_marriages"], errors="coerce"),
            "marital_status_cumulative": pd.to_numeric(
                raw["marital_status_cumulative"], errors="coerce"
            ),
            "ui_spells": pd.to_numeric(raw["ui_spells_2021"], errors="coerce").fillna(
                pd.to_numeric(raw["ui_spells_2019"], errors="coerce")
            ),
            "first_marriage_age": pd.to_numeric(
                raw["first_marriage_year"], errors="coerce"
            ) - pd.to_numeric(raw["birth_year"], errors="coerce"),
            "first_child_age": pd.to_numeric(
                raw["first_child_birth_year_2019"], errors="coerce"
            ) - pd.to_numeric(raw["birth_year"], errors="coerce"),
            "sat_math_bin": _clean_sentinel(raw["sat_math_2007_bin"]).astype("Int64").astype(str),
            "sat_verbal_bin": _clean_sentinel(raw["sat_verbal_2007_bin"]).astype("Int64").astype(str),
            "act_bin": _clean_sentinel(raw["act_2007_bin"]).astype("Int64").astype(str),
        }
    )
    for col in SAT_ACT_COLS:
        extra.loc[extra[col] == "<NA>", col] = pd.NA
    df = std.merge(extra, on="person_id", how="left")
    ever_married = (
        pd.to_numeric(df["num_marriages"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(df["marital_status_cumulative"], errors="coerce").fillna(0).gt(0)
    )
    df["ever_married"] = ever_married.astype(int)
    df["log_household_income"] = _positive_log(df["household_income"])
    df["signed_log_net_worth"] = _signed_log1p(df["net_worth"])
    df["log_annual_earnings_real"] = _positive_log(df["annual_earnings_real"])
    return df


def _run_deep_models(
    df: pd.DataFrame,
    blocks: dict[str, list[str]],
) -> pd.DataFrame:
    prepared = df.copy()
    prepared["age_sq"] = prepared["age"] ** 2
    prepared["person_weight"] = pd.to_numeric(prepared["person_weight"], errors="coerce").fillna(1.0)
    full_model = list(blocks.values())[-1]
    required = ["log_annual_earnings_real", "person_weight"] + design_source_columns(full_model)
    common_mask = prepared["log_annual_earnings_real"].notna() & prepared["person_weight"].gt(0)
    for column in required:
        if column in {"log_annual_earnings_real", "person_weight"}:
            continue
        common_mask &= prepared[column].notna()
    prepared = prepared.loc[common_mask].copy()
    results = run_sequential_ols(
        prepared,
        outcome="log_annual_earnings_real",
        weight_col="person_weight",
        blocks=blocks,
    )
    summary = results_to_dataframe(results)
    summary["pct_gap"] = summary["female_coef"].apply(_pct_from_log_coef)
    return summary


def _blocks_for_nlsy79(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
    labels = {
        "D0": "female only",
        "D1": "demographics and education",
        "D2": "skills and noncognitive traits",
        "D3": "family background",
        "D4": "occupation sorting",
        "D5": "family formation",
        "D6": "adult resources",
    }
    blocks = {
        "D0": ["female"],
        "D1": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
        "D2": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "rotter_locus_control",
            "self_esteem",
        ],
        "D3": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "rotter_locus_control",
            "self_esteem",
            "parent_education",
            "mother_education",
            "father_education",
        ],
        "D4": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "rotter_locus_control",
            "self_esteem",
            "parent_education",
            "mother_education",
            "father_education",
            "C(occupation_code)",
        ],
        "D5": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "rotter_locus_control",
            "self_esteem",
            "parent_education",
            "mother_education",
            "father_education",
            "C(occupation_code)",
            "C(marital_status)",
            "ever_married",
            "number_children",
            "age_first_birth",
            "age_first_marriage",
        ],
        "D6": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "rotter_locus_control",
            "self_esteem",
            "parent_education",
            "mother_education",
            "father_education",
            "C(occupation_code)",
            "C(marital_status)",
            "ever_married",
            "number_children",
            "age_first_birth",
            "age_first_marriage",
            "log_household_income",
            "signed_log_net_worth",
            "homeowner",
        ],
    }
    return {k: _available_controls(df, v) for k, v in blocks.items()}, labels


def _blocks_for_nlsy97(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
    labels = {
        "D0": "female only",
        "D1": "demographics and education",
        "D2": "skills and school achievement",
        "D3": "family background",
        "D4": "occupation sorting",
        "D5": "family formation",
        "D6": "adult resources",
    }
    blocks = {
        "D0": ["female"],
        "D1": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
        "D2": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "C(sat_math_bin)",
            "C(sat_verbal_bin)",
            "C(act_bin)",
        ],
        "D3": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "C(sat_math_bin)",
            "C(sat_verbal_bin)",
            "C(act_bin)",
            "parent_education",
            "mother_education",
            "father_education",
        ],
        "D4": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "C(sat_math_bin)",
            "C(sat_verbal_bin)",
            "C(act_bin)",
            "parent_education",
            "mother_education",
            "father_education",
            "C(occupation_code)",
        ],
        "D5": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "C(sat_math_bin)",
            "C(sat_verbal_bin)",
            "C(act_bin)",
            "parent_education",
            "mother_education",
            "father_education",
            "C(occupation_code)",
            "ever_married",
            "num_marriages",
            "number_children",
            "first_child_age",
            "first_marriage_age",
        ],
        "D6": [
            "female",
            "age",
            "age_sq",
            "C(race_ethnicity)",
            "C(education_level)",
            "g_proxy",
            "C(sat_math_bin)",
            "C(sat_verbal_bin)",
            "C(act_bin)",
            "parent_education",
            "mother_education",
            "father_education",
            "C(occupation_code)",
            "ever_married",
            "num_marriages",
            "number_children",
            "first_child_age",
            "first_marriage_age",
            "log_household_income",
            "signed_log_net_worth",
            "ui_spells",
        ],
    }
    return {k: _available_controls(df, v) for k, v in blocks.items()}, labels


def _cohort_summary(
    dataset: str,
    summary: pd.DataFrame,
    transitions: pd.DataFrame,
) -> dict[str, object]:
    pct = summary.set_index("model")["pct_gap"]
    reduction_by_to = transitions.set_index("to_model")["reduction_pp"]
    biggest = transitions.sort_values("reduction_pp", ascending=False).iloc[0]
    return {
        "dataset": dataset,
        "raw_gap_pct": float(pct["D0"]),
        "demographics_gap_pct": float(pct["D1"]),
        "skills_gap_pct": float(pct["D2"]),
        "background_gap_pct": float(pct["D3"]),
        "occupation_gap_pct": float(pct["D4"]),
        "family_gap_pct": float(pct["D5"]),
        "resources_gap_pct": float(pct["D6"]),
        "skills_reduction_pp": float(reduction_by_to.get("D2", np.nan)),
        "background_reduction_pp": float(reduction_by_to.get("D3", np.nan)),
        "occupation_reduction_pp": float(reduction_by_to.get("D4", np.nan)),
        "family_reduction_pp": float(reduction_by_to.get("D5", np.nan)),
        "resources_reduction_pp": float(reduction_by_to.get("D6", np.nan)),
        "largest_reduction_block": str(biggest["added_block"]),
        "largest_reduction_pp": float(biggest["reduction_pp"]),
        "final_gap_pct": float(pct["D6"]),
        "common_sample_n": int(summary.loc[summary["model"] == "D6", "n_obs"].iloc[0]),
    }


def build_report(
    cohort_comparison: pd.DataFrame,
    factor_contrib: pd.DataFrame,
) -> str:
    indexed = cohort_comparison.set_index("dataset")
    lines = [
        "# NLSY Deep Dive",
        "",
        "This note extends the NLSY lane beyond the older `g_proxy`-only framing. The cohort models now add blocks for skills and traits, family background, occupation sorting, family formation, and adult resources.",
        "",
        "Important interpretation rule: these are not all exogenous causal controls. Some later-life blocks, especially family formation and adult resources, are better read as mechanism-sensitive accounting blocks than as clean pre-market controls.",
        "All block comparisons below use the common complete-case sample for the cohort's final model, so the reductions are not being driven by stepwise sample changes.",
        "",
    ]
    for dataset in ["NLSY79", "NLSY97"]:
        row = indexed.loc[dataset]
        top_blocks = factor_contrib[factor_contrib["dataset"] == dataset].sort_values(
            "reduction_pp", ascending=False
        )
        first = top_blocks.iloc[0]
        second = top_blocks.iloc[1]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Raw annual-earnings gap: {row['raw_gap_pct']:.2f}%",
                f"- Final deep-model gap: {row['final_gap_pct']:.2f}%",
                f"- Common-sample observations: {int(row['common_sample_n'])}",
                f"- Largest reduction comes from: {row['largest_reduction_block']} ({row['largest_reduction_pp']:.2f} percentage points).",
                f"- Next-largest reduction: {second['added_block']} ({second['reduction_pp']:.2f} points).",
                f"- Skills block reduction: {row['skills_reduction_pp']:.2f} points.",
                f"- Family-background block reduction: {row['background_reduction_pp']:.2f} points.",
                f"- Occupation-sorting block reduction: {row['occupation_reduction_pp']:.2f} points.",
                f"- Family-formation block reduction: {row['family_reduction_pp']:.2f} points.",
                f"- Adult-resources block reduction: {row['resources_reduction_pp']:.2f} points.",
                "- Caution: the adult-resources block is post-market and should be read as mechanism-sensitive accounting, not a clean exogenous explanation.",
                "",
            ]
        )
    lines.extend(
        [
            "## Takeaway",
            "",
            "NLSY is useful because it can test richer background and life-course factors than ACS or CPS. The main lesson from these cohort files is still not that one factor explains the gap. Skills, background, occupation, and family/resource variables each matter somewhat, but meaningful residual gaps remain after all of them are added.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    n79 = _prepare_nlsy79()
    blocks79, labels79 = _blocks_for_nlsy79(n79)
    summary79 = _run_deep_models(n79, blocks79)
    transitions79 = pd.DataFrame(_make_transition_rows("NLSY79", "1979 cohort", summary79, labels79))

    n97 = _prepare_nlsy97()
    blocks97, labels97 = _blocks_for_nlsy97(n97)
    summary97 = _run_deep_models(n97, blocks97)
    transitions97 = pd.DataFrame(_make_transition_rows("NLSY97", "1997 cohort", summary97, labels97))

    factor_contrib = pd.concat([transitions79, transitions97], ignore_index=True)
    cohort_comparison = pd.DataFrame(
        [
            _cohort_summary("NLSY79", summary79, transitions79),
            _cohort_summary("NLSY97", summary97, transitions97),
        ]
    )

    factor_contrib.to_csv(DIAG_DIR / "nlsy_factor_contributions.csv", index=False)
    cohort_comparison.to_csv(DIAG_DIR / "nlsy_cohort_comparison.csv", index=False)

    report = build_report(cohort_comparison, factor_contrib)
    (REPORTS_DIR / "nlsy_deep_dive.md").write_text(report + "\n", encoding="utf-8")

    print(f"Wrote {DIAG_DIR / 'nlsy_factor_contributions.csv'}")
    print(f"Wrote {DIAG_DIR / 'nlsy_cohort_comparison.csv'}")
    print(f"Wrote {REPORTS_DIR / 'nlsy_deep_dive.md'}")


if __name__ == "__main__":
    main()
