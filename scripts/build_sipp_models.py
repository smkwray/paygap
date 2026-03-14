#!/usr/bin/env python3
"""Build a modest SIPP-specific adjusted-gap surface from the validated 2023 file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gender_gap.models.descriptive import raw_gap
from gender_gap.models.ols import results_to_dataframe, run_sequential_ols


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sipp_standardized.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "sipp" / "2023"
REPORTS_DIR = PROJECT_ROOT / "reports"

SIPP_BLOCKS = {
    "SIPP0": ["female"],
    "SIPP1": ["female", "C(month)"],
    "SIPP2": ["female", "C(month)", "C(occupation_code)", "C(industry_code)"],
    "SIPP3": [
        "female",
        "C(month)",
        "C(occupation_code)",
        "C(industry_code)",
        "usual_hours_week",
        "paid_hourly",
        "multiple_jobholder",
    ],
}


def _pct_from_log_coef(value: float) -> float:
    return (np.exp(value) - 1.0) * 100.0


def build_report(ols_df: pd.DataFrame, raw_gap_pct: float, path: Path) -> Path:
    latest = ols_df.iloc[-1]
    lines = [
        "# SIPP Models",
        "",
        "This note adds a modest adjusted-gap surface for the validated 2023 public-use SIPP file.",
        "The control sequence only uses variables that are actually present and well-covered in the current standardized SIPP output.",
        "",
        "## Specification ladder",
        "",
        "- `SIPP0`: female only",
        "- `SIPP1`: + month",
        "- `SIPP2`: + occupation and industry",
        "- `SIPP3`: + usual hours, paid-hourly status, multiple-jobholder indicator",
        "",
        "## Headline results",
        "",
        f"- Raw hourly wage gap: {raw_gap_pct:.2f}%",
        f"- Latest adjusted log-point gap (`SIPP3`): {latest['female_coef']:.4f}",
        f"- Latest adjusted percent gap (`SIPP3`): {latest['pct_gap']:.2f}%",
        f"- Latest adjusted model R² (`SIPP3`): {latest['r_squared']:.4f}",
        f"- Worker observations in `SIPP3`: {int(latest['n_obs']):,}",
        "",
        "## Interpretation",
        "",
        "- This is a SIPP-specific adjustment surface, not a substitute for the richer ACS sequential models.",
        "- The main value is to show whether the 2023 SIPP worker gap remains material after conditioning on month, job sorting, and basic job-structure variables.",
        "- If SIPP is extended further, the next honest step is richer covariate recovery from the public-use release rather than simply adding more models on the same limited feature set.",
        "",
        "## Model table",
        "",
        "| Model | Female coef | Percent gap | R² | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ols_df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.female_coef:.4f} | {row.pct_gap:.2f}% | {row.r_squared:.4f} | {int(row.n_obs):,} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)
    workers = df[
        df["employed"].fillna(0).eq(1)
        & df["hourly_wage_real"].gt(0)
        & df["hourly_wage_real"].notna()
        & df["person_weight"].gt(0)
    ].copy()

    raw = raw_gap(workers, outcome="hourly_wage_real", weight="person_weight")
    ols = run_sequential_ols(
        workers,
        outcome="log_hourly_wage_real",
        weight_col="person_weight",
        blocks=SIPP_BLOCKS,
    )
    ols_df = results_to_dataframe(ols)
    ols_df["pct_gap"] = ols_df["female_coef"].apply(_pct_from_log_coef)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([raw]).to_csv(RESULTS_DIR / "raw_gap.csv", index=False)
    ols_df.to_csv(RESULTS_DIR / "ols_sequential.csv", index=False)
    build_report(ols_df, float(raw["gap_pct"]), REPORTS_DIR / "sipp_models.md")

    print(f"Wrote {RESULTS_DIR / 'raw_gap.csv'}")
    print(f"Wrote {RESULTS_DIR / 'ols_sequential.csv'}")
    print(f"Wrote {REPORTS_DIR / 'sipp_models.md'}")


if __name__ == "__main__":
    main()
