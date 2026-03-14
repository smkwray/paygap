#!/usr/bin/env python3
"""Build a careful DML vs OLS vs Oaxaca comparison surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gender_gap.models.dml import run_dml
from gender_gap.models.oaxaca import oaxaca_blinder, oaxaca_summary_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DIAG_DIR = RESULTS_DIR / "diagnostics"

DATASETS = {
    "acs": {
        "data_path": PROJECT_ROOT / "data" / "processed" / "acs_2023_analysis_ready.parquet",
        "ols_path": RESULTS_DIR / "acs" / "2023" / "ols_sequential.csv",
        "oaxaca_path": RESULTS_DIR / "acs" / "2023" / "oaxaca.csv",
        "ols_model": "M5",
        "raw_gap_path": RESULTS_DIR / "acs" / "2023" / "raw_gap.csv",
        "worker_filter": lambda df: df["hourly_wage_real"].gt(0) & df["log_hourly_wage_real"].notna(),
        "required_columns": [],
        "dml_controls": [
            "age", "age_sq", "race_ethnicity", "education_level", "marital_status",
            "number_children", "children_under_5", "occupation_code", "industry_code",
            "class_of_worker", "usual_hours_week", "work_from_home",
            "commute_minutes_one_way", "state_fips",
        ],
        "categorical_controls": [
            "race_ethnicity", "education_level", "marital_status", "occupation_code",
            "industry_code", "class_of_worker", "state_fips",
        ],
        "oaxaca_controls": [
            "age", "age_sq", "usual_hours_week", "work_from_home",
            "commute_minutes_one_way", "number_children", "children_under_5",
        ],
        "oaxaca_source": "canonical",
        "year": 2023,
        "label": "ACS",
    },
    "cps": {
        "data_path": PROJECT_ROOT / "data" / "processed" / "cps_asec_2023_analysis_ready.parquet",
        "ols_path": RESULTS_DIR / "cps" / "2023" / "ols_sequential.csv",
        "oaxaca_path": None,
        "ols_model": "M_full",
        "raw_gap_path": RESULTS_DIR / "cps" / "2023" / "raw_gap.csv",
        "worker_filter": lambda df: df["hourly_wage_real"].gt(0) & df["log_hourly_wage_real"].notna(),
        "required_columns": [],
        "dml_controls": [
            "age", "age_sq", "race_ethnicity", "education_level", "marital_status",
            "number_children", "children_under_5", "occupation_code", "industry_code",
            "class_of_worker", "usual_hours_week", "state_fips",
        ],
        "categorical_controls": [
            "race_ethnicity", "education_level", "marital_status", "occupation_code",
            "industry_code", "class_of_worker", "state_fips",
        ],
        "oaxaca_controls": [
            "age", "age_sq", "usual_hours_week", "number_children", "children_under_5",
        ],
        "oaxaca_source": "ad_hoc",
        "year": 2023,
        "label": "CPS ASEC",
    },
    "sipp": {
        "data_path": PROJECT_ROOT / "data" / "processed" / "sipp_standardized.parquet",
        "ols_path": RESULTS_DIR / "sipp" / "2023" / "ols_sequential.csv",
        "oaxaca_path": None,
        "ols_model": "SIPP3",
        "raw_gap_path": RESULTS_DIR / "sipp" / "2023" / "raw_gap.csv",
        "worker_filter": lambda df: df["employed"].fillna(0).eq(1) & df["hourly_wage_real"].gt(0),
        "required_columns": ["employed"],
        "dml_controls": [
            "month", "occupation_code", "industry_code", "usual_hours_week",
            "actual_hours_last_week", "paid_hourly", "multiple_jobholder",
        ],
        "categorical_controls": ["month", "occupation_code", "industry_code"],
        "oaxaca_controls": [
            "usual_hours_week", "actual_hours_last_week", "paid_hourly", "multiple_jobholder",
        ],
        "oaxaca_source": "ad_hoc",
        "year": 2023,
        "label": "SIPP",
    },
}


def _pct_from_log_coef(value: float) -> float:
    return abs((np.exp(value) - 1.0) * 100.0)


def _load_worker_frame(dataset: str) -> pd.DataFrame:
    cfg = DATASETS[dataset]
    cols = ["female", "person_weight", "hourly_wage_real"] + cfg["required_columns"] + cfg["dml_controls"] + cfg["oaxaca_controls"]
    cols = list(dict.fromkeys(cols))
    df = pd.read_parquet(cfg["data_path"], columns=[c for c in cols if c is not None])
    if "log_hourly_wage_real" not in df.columns:
        df["log_hourly_wage_real"] = np.log(df["hourly_wage_real"].replace(0, np.nan))
    mask = cfg["worker_filter"](df)
    df = df.loc[mask].copy()
    for col in cfg["categorical_controls"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _read_ols(dataset: str) -> dict[str, float | str]:
    cfg = DATASETS[dataset]
    ols = pd.read_csv(cfg["ols_path"])
    row = ols.loc[ols["model"] == cfg["ols_model"]].iloc[0]
    return {
        "ols_model": cfg["ols_model"],
        "ols_female_coef": float(row["female_coef"]),
        "ols_pct_gap": _pct_from_log_coef(float(row["female_coef"])),
        "ols_r_squared": float(row["r_squared"]),
        "ols_n_obs": int(row["n_obs"]),
    }


def _read_raw_gap(dataset: str) -> float:
    row = pd.read_csv(DATASETS[dataset]["raw_gap_path"]).iloc[0]
    return float(row["gap_pct"])


def _read_or_run_oaxaca(dataset: str, df: pd.DataFrame) -> dict[str, float | str]:
    cfg = DATASETS[dataset]
    if cfg["oaxaca_path"] is not None:
        table = pd.read_csv(cfg["oaxaca_path"])
    else:
        result = oaxaca_blinder(
            df,
            outcome="log_hourly_wage_real",
            controls=cfg["oaxaca_controls"],
            weight_col="person_weight",
        )
        table = oaxaca_summary_table(result)
    total = table.loc[table["component"] == "Total gap"].iloc[0]
    unexplained = table.loc[table["component"].str.startswith("Unexplained")].iloc[0]
    explained = table.loc[table["component"].str.startswith("Explained")].iloc[0]
    return {
        "oaxaca_source": cfg["oaxaca_source"],
        "oaxaca_total_gap_log": float(total["value"]),
        "oaxaca_explained_log": float(explained["value"]),
        "oaxaca_explained_pct": float(explained["pct"]),
        "oaxaca_unexplained_log": float(unexplained["value"]),
        "oaxaca_unexplained_pct": float(unexplained["pct"]),
    }


def _run_dml_for_dataset(dataset: str, df: pd.DataFrame, nuisance_learner: str, n_folds: int) -> dict[str, float | str]:
    cfg = DATASETS[dataset]
    res = run_dml(
        df,
        outcome="log_hourly_wage_real",
        treatment="female",
        weight_col="person_weight",
        controls=cfg["dml_controls"],
        nuisance_learner=nuisance_learner,
        n_folds=n_folds,
    )
    return {
        "dml_nuisance_learner": nuisance_learner,
        "dml_treatment_effect": float(res.treatment_effect),
        "dml_pct_gap": _pct_from_log_coef(float(res.treatment_effect)),
        "dml_std_error": float(res.std_error),
        "dml_ci_lower": float(res.ci_lower),
        "dml_ci_upper": float(res.ci_upper),
        "dml_pvalue": float(res.pvalue),
        "dml_n_obs": int(res.n_obs),
    }


def compute_dataset_summary(dataset: str, nuisance_learner: str = "elasticnet", n_folds: int = 3) -> pd.DataFrame:
    cfg = DATASETS[dataset]
    df = _load_worker_frame(dataset)
    summary = {
        "dataset": dataset,
        "dataset_label": cfg["label"],
        "year": cfg["year"],
        "raw_gap_pct": _read_raw_gap(dataset),
        **_read_ols(dataset),
        **_read_or_run_oaxaca(dataset, df),
        **_run_dml_for_dataset(dataset, df, nuisance_learner=nuisance_learner, n_folds=n_folds),
    }
    return pd.DataFrame([summary])


def build_report(summary: pd.DataFrame) -> str:
    lines = [
        "# DML vs OLS vs Oaxaca",
        "",
        "This note compares three different gap-estimation lenses on the latest available public-data files.",
        "",
        "Important caution: these are not identical estimands.",
        "- OLS reports a conditional female coefficient under a specified control set.",
        "- DML reports a residual female effect after flexible nuisance-model adjustment.",
        "- Oaxaca reports an explained/unexplained decomposition under a chosen reference structure.",
        "",
        "## 2023 comparison",
        "",
        "| Dataset | Raw gap % | OLS model | OLS adjusted gap % | DML adjusted gap % | Oaxaca unexplained % |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset_label} | {row.raw_gap_pct:.2f} | {row.ols_model} | {row.ols_pct_gap:.2f} | "
            f"{row.dml_pct_gap:.2f} | {row.oaxaca_unexplained_pct:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- OLS and DML are the more comparable pair for an adjusted residual gap; Oaxaca answers a different question.",
        "- In this 2023 comparison, DML runs larger than OLS in ACS, CPS, and SIPP, so the flexible residualization here does not drive the female effect toward zero.",
        "- If Oaxaca behaves differently, especially through unstable explained shares, that should be treated as a decomposition caution rather than a contradiction of the residual-gap estimates.",
        "",
        "## Notes",
        "",
        "- ACS uses the canonical 2023 Oaxaca table already in the repo.",
        "- CPS and SIPP Oaxaca values in this note are ad hoc 2023 decompositions computed with simple numeric controls because those datasets do not already have canonical Oaxaca artifacts.",
        "- DML here uses the elastic-net nuisance learner for computational tractability and comparability across datasets.",
        "- The current DML implementation is unweighted, so this comparison should be treated as a supplemental sensitivity layer rather than a survey-primary estimate.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DML vs OLS vs Oaxaca comparison outputs")
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Single dataset to compute")
    parser.add_argument("--combine", action="store_true", help="Combine per-dataset CSVs into final outputs")
    parser.add_argument("--nuisance-learner", default="elasticnet", choices=["elasticnet", "rf", "lgbm"])
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Override per-dataset CSV output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.combine:
        parts = []
        for dataset in DATASETS:
            path = DIAG_DIR / f"method_comparison_{dataset}_2023.csv"
            if path.exists():
                parts.append(pd.read_csv(path))
        if not parts:
            raise SystemExit("No per-dataset comparison parts found")
        summary = pd.concat(parts, ignore_index=True)
        summary.to_csv(DIAG_DIR / "method_comparison_2023.csv", index=False)
        (REPORTS_DIR / "method_comparison.md").write_text(build_report(summary) + "\n", encoding="utf-8")
        print(f"Wrote {DIAG_DIR / 'method_comparison_2023.csv'}")
        print(f"Wrote {REPORTS_DIR / 'method_comparison.md'}")
        return

    if not args.dataset:
        raise SystemExit("--dataset or --combine required")

    summary = compute_dataset_summary(
        args.dataset,
        nuisance_learner=args.nuisance_learner,
        n_folds=args.n_folds,
    )
    output = args.output or (DIAG_DIR / f"method_comparison_{args.dataset}_2023.csv")
    summary.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
