#!/usr/bin/env python3
"""Build ACS employment-selection robustness outputs by year."""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from run_all_analyses import DATA_RAW, RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("build_acs_selection_outputs")

DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
ACS_SELECTION_RAW_COLUMNS = [
    "SERIALNO",
    "SPORDER",
    "SEX",
    "AGEP",
    "SCHL",
    "MAR",
    "HISP",
    "RAC1P",
    "OCCP",
    "INDP",
    "COW",
    "WKHP",
    "WKW",
    "WKWN",
    "JWTR",
    "JWTRNS",
    "JWMNP",
    "ST",
    "PUMA",
    "POWSP",
    "POWPUMA",
    "NOC",
    "PAOC",
    "WAGP",
    "PERNP",
    "ADJINC",
    "PWGTP",
    "ESR",
]


def _available_raw_columns(path: Path) -> list[str]:
    schema_cols = set(pq.read_schema(path).names)
    return [col for col in ACS_SELECTION_RAW_COLUMNS if col in schema_cols]


def prepare_acs_selection_sample(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Prepare a prime-age civilian ACS sample for selection robustness."""
    from gender_gap.standardize.acs_standardize import standardize_acs

    df = standardize_acs(df_raw, survey_year=year).copy()
    df["employed"] = df_raw["ESR"].isin([1, 2]).astype(int)
    labor_force_status = pd.Series("not_in_labor_force", index=df.index)
    labor_force_status[df_raw["ESR"].isin([1, 2]).to_numpy()] = "employed"
    labor_force_status[df_raw["ESR"].eq(3).to_numpy()] = "unemployed"
    df["labor_force_status"] = labor_force_status
    df["esr"] = pd.to_numeric(df_raw["ESR"], errors="coerce").to_numpy()

    mask = (
        df["age"].between(25, 54)
        & (df["person_weight"] > 0)
        & ~df["esr"].isin([4, 5])
    )
    df = df.loc[mask].copy()
    logger.info("ACS %d: %d selection-sample rows", year, len(df))
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write ACS selection-robustness outputs to results/acs/<year>/.",
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--raw-variant",
        choices=["api", "api_repweights"],
        default="api",
        help="Which ACS raw files to use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite selection_robustness.csv even if it already exists.",
    )
    args = parser.parse_args(argv)

    from gender_gap.models.selection import run_selection_robustness

    suffix = "_api_repweights" if args.raw_variant == "api_repweights" else "_api"
    for year in args.years:
        raw_path = DATA_RAW / "acs" / f"acs_pums_{year}{suffix}.parquet"
        if not raw_path.exists():
            logger.warning("ACS %d raw file missing: %s", year, raw_path)
            continue

        output_dir = RESULTS_DIR / "acs" / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "selection_robustness.csv"
        if output_path.exists() and not args.overwrite:
            logger.info("ACS %d selection output already exists: %s", year, output_path)
            continue

        df_raw = pd.read_parquet(
            raw_path,
            columns=_available_raw_columns(raw_path),
            filters=[
                ("AGEP", ">=", 25),
                ("AGEP", "<=", 54),
                ("PWGTP", ">", 0),
            ],
        )
        if "ESR" not in df_raw.columns:
            logger.warning("ACS %d raw file lacks ESR: %s", year, raw_path)
            continue

        selection_df = prepare_acs_selection_sample(df_raw, year)
        result = run_selection_robustness(selection_df)
        result.to_csv(output_path, index=False)
        logger.info("ACS %d selection output written: %s", year, output_path)
        del df_raw, selection_df, result
        gc.collect()


if __name__ == "__main__":
    main()
