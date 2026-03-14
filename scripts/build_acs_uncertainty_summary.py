#!/usr/bin/env python3
"""Build ACS replicate-weight uncertainty summaries from processed parquet files."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIAGNOSTICS = PROJECT_ROOT / "results" / "diagnostics"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gender_gap.models.descriptive import raw_gap_with_sdr  # noqa: E402
from gender_gap.models.ols import female_coefficient_with_sdr, required_columns_for_model  # noqa: E402
from gender_gap.utils.weights import replicate_weight_columns  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("acs_uncertainty")


def _available_years() -> list[int]:
    years = []
    for path in DATA_PROCESSED.glob("acs_*_analysis_ready_repweights.parquet"):
        years.append(int(path.name.split("_")[1]))
    return sorted(set(years))


def _schema_columns(path: Path) -> list[str]:
    return pq.read_schema(path).names


def _read_subset(path: Path, columns: list[str]) -> pd.DataFrame:
    seen: set[str] = set()
    subset = [col for col in columns if not (col in seen or seen.add(col))]
    logger.info(
        "ACS uncertainty %s: reading %d columns",
        path.name,
        len(subset),
    )
    return pd.read_parquet(path, columns=subset)


def build_summary_for_year(
    year: int,
    model_name: str = "M5",
    metrics: tuple[str, ...] = ("raw_gap_pct", "model"),
) -> pd.DataFrame:
    path = DATA_PROCESSED / f"acs_{year}_analysis_ready_repweights.parquet"
    logger.info("ACS uncertainty %d: preparing %s", year, path.name)
    schema_columns = _schema_columns(path)
    rep_cols = replicate_weight_columns(schema_columns, prefix="PWGTP")
    if not rep_cols:
        raise ValueError(f"ACS uncertainty {year}: no ACS replicate weights found")

    rows = []
    if "raw_gap_pct" in metrics:
        raw_cols = ["female", "hourly_wage_real", "person_weight", *rep_cols]
        df = _read_subset(path, raw_cols)
        raw = raw_gap_with_sdr(df)
        rows.append({
            "year": year,
            "metric": "raw_gap_pct",
            "estimate": raw["gap_pct"],
            "se": raw["gap_pct_se"],
            "moe90": raw["gap_pct_moe90"],
            "ci90_low": raw["gap_pct_ci90_low"],
            "ci90_high": raw["gap_pct_ci90_high"],
            "ci95_low": raw["gap_pct_ci95_low"],
            "ci95_high": raw["gap_pct_ci95_high"],
            "n_obs": raw["n_male"] + raw["n_female"],
            "n_replicates": raw["n_replicates"],
        })
        logger.info("ACS uncertainty %d: raw gap %.2f%%", year, raw["gap_pct"])
        del df
        gc.collect()

    if "model" in metrics:
        model_cols = required_columns_for_model(
            model_name=model_name,
            outcome="log_hourly_wage_real",
            weight_col="person_weight",
        )
        df = _read_subset(path, [*model_cols, *rep_cols])
        ols = female_coefficient_with_sdr(df, model_name=model_name)
        rows.append({
            "year": year,
            "metric": f"{model_name}_female_coef",
            "estimate": ols["female_coef"],
            "se": ols["female_coef_sdr_se"],
            "moe90": ols["female_coef_moe90"],
            "ci90_low": ols["female_coef_ci90_low"],
            "ci90_high": ols["female_coef_ci90_high"],
            "ci95_low": ols["female_coef_ci95_low"],
            "ci95_high": ols["female_coef_ci95_high"],
            "n_obs": ols["n_obs"],
            "n_replicates": ols["n_replicates"],
        })
        logger.info("ACS uncertainty %d: %s %.4f", year, model_name, ols["female_coef"])
        del df
        gc.collect()

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build ACS SDR uncertainty summary from repweight-aware processed files.",
    )
    parser.add_argument("--years", nargs="+", type=int, help="Years to summarize")
    parser.add_argument("--model", default="M5", help="OLS model name to summarize")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["raw_gap_pct", "model"],
        default=["raw_gap_pct", "model"],
        help="Which uncertainty metrics to build",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild years even if they already exist in the output summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output CSV path. Defaults to results/diagnostics/acs_uncertainty_summary.csv",
    )
    args = parser.parse_args(argv)

    years = args.years or _available_years()
    if not years:
        raise SystemExit("No acs_*_analysis_ready_repweights.parquet files found")

    RESULTS_DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (RESULTS_DIAGNOSTICS / "acs_uncertainty_summary.csv")
    existing = pd.DataFrame()
    if out_path.exists():
        existing = pd.read_csv(out_path)

    done_years: set[int] = set()
    if not existing.empty and not args.force:
        done_years = set(existing["year"].unique().tolist())

    metrics = tuple(args.metrics)
    metric_names = {"raw_gap_pct"}
    if "model" in metrics:
        metric_names.add(f"{args.model}_female_coef")

    pending_years = []
    for year in years:
        if args.force or existing.empty:
            pending_years.append(year)
            continue
        year_metrics = set(existing.loc[existing["year"] == year, "metric"].tolist())
        if not metric_names.issubset(year_metrics):
            pending_years.append(year)
    if not pending_years:
        logger.info("ACS uncertainty: all requested years already present in %s", out_path)
        print(f"Wrote {out_path}")
        return

    frames = [existing] if not existing.empty else []
    for year in pending_years:
        if not existing.empty and not args.force:
            existing = existing.loc[existing["year"] != year].copy()
            frames = [existing] + [frame for frame in frames[1:] if not frame.empty]
        frames.append(build_summary_for_year(year, model_name=args.model, metrics=metrics))
        summary = pd.concat(frames, ignore_index=True).sort_values(["year", "metric"])
        summary.to_csv(out_path, index=False)
        logger.info("ACS uncertainty %d: checkpoint written to %s", year, out_path)

    summary = pd.concat(frames, ignore_index=True).sort_values(["year", "metric"])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
