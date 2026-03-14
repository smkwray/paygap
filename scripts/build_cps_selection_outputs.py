#!/usr/bin/env python3
"""Build CPS ASEC employment-selection robustness outputs by year."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from run_all_analyses import (
    CPS_NUMERIC_COLS,
    DATA_RAW,
    RESULTS_DIR,
    prepare_cps_asec_selection_sample,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("build_cps_selection_outputs")

DEFAULT_YEARS = list(range(2015, 2024))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write CPS ASEC selection-robustness outputs to results/cps/<year>/.",
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite selection_robustness.csv even if it already exists.",
    )
    args = parser.parse_args(argv)

    from gender_gap.models.selection import run_selection_robustness

    for year in args.years:
        raw_path = DATA_RAW / "cps" / f"cps_asec_{year}_api.parquet"
        if not raw_path.exists():
            logger.warning("CPS %d raw file missing: %s", year, raw_path)
            continue

        output_dir = RESULTS_DIR / "cps" / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "selection_robustness.csv"
        if output_path.exists() and not args.overwrite:
            logger.info("CPS %d selection output already exists: %s", year, output_path)
            continue

        df_raw = pd.read_parquet(raw_path)
        for col in CPS_NUMERIC_COLS:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        selection_df = prepare_cps_asec_selection_sample(df_raw, year)
        if selection_df is None or len(selection_df) == 0:
            logger.warning("CPS %d selection sample empty", year)
            continue

        result = run_selection_robustness(selection_df)
        result.to_csv(output_path, index=False)
        logger.info("CPS %d selection output written: %s", year, output_path)


if __name__ == "__main__":
    main()
