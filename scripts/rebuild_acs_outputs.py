#!/usr/bin/env python3
"""Rebuild ACS processed/model outputs from a selected raw variant.

This is intended for targeted ACS reruns after upstream raw-file fixes, such as
the family-variable backfill, without rerunning the full cross-dataset stack.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_all_analyses import (  # noqa: E402
    compile_trend_results,
    run_acs_models_single_year,
    run_pooled_acs_analysis,
    standardize_acs_year,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rebuild_acs_outputs")

DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]


def _processed_path(year: int, raw_variant: str) -> Path:
    suffix = "_repweights" if raw_variant == "api_repweights" else ""
    return DATA_PROCESSED / f"acs_{year}_analysis_ready{suffix}.parquet"


def _year_results_path(year: int) -> Path:
    return RESULTS_DIR / "acs" / str(year)


def _remove_existing_year_artifacts(year: int, raw_variant: str) -> None:
    processed_path = _processed_path(year, raw_variant)
    if processed_path.exists():
        processed_path.unlink()
        logger.info("Removed %s", processed_path)

    year_dir = _year_results_path(year)
    if year_dir.exists():
        shutil.rmtree(year_dir)
        logger.info("Removed %s", year_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild ACS processed/model outputs from existing raw files.",
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--raw-variant",
        choices=["api", "api_repweights"],
        default="api",
        help="Which ACS raw files to standardize from.",
    )
    parser.add_argument(
        "--rebuild-pooled",
        action="store_true",
        help="Also rebuild pooled ACS outputs after the year-by-year run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing processed/year results for the requested years first.",
    )
    args = parser.parse_args(argv)

    results_by_year: dict[int, dict] = {}
    data_by_year: dict[int, pd.DataFrame] = {}

    for year in args.years:
        if args.overwrite:
            _remove_existing_year_artifacts(year, args.raw_variant)

        df = standardize_acs_year(year, raw_variant=args.raw_variant)
        if df is None or len(df) == 0:
            logger.warning("ACS %d: no rebuild output", year)
            continue

        data_by_year[year] = df
        results_by_year[year] = run_acs_models_single_year(
            df,
            year,
            _year_results_path(year),
        )

    if results_by_year:
        compile_trend_results(results_by_year, "acs", RESULTS_DIR / "trends")

    if args.rebuild_pooled:
        if args.overwrite:
            pooled_dir = RESULTS_DIR / "acs_pooled"
            if pooled_dir.exists():
                shutil.rmtree(pooled_dir)
                logger.info("Removed %s", pooled_dir)
        run_pooled_acs_analysis(data_by_year, RESULTS_DIR / "acs_pooled")


if __name__ == "__main__":
    main()
