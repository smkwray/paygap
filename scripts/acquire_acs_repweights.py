#!/usr/bin/env python3
"""Acquire ACS API extracts with replicate weights and audit the result."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "acs"
RESULTS_DIAGNOSTICS = PROJECT_ROOT / "results" / "diagnostics"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gender_gap.downloaders.acs import ACSDownloader  # noqa: E402
from run_all_analyses import standardize_acs_year  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]


def _display_path(path: Path) -> str:
    """Render a project-relative path when possible, otherwise absolute."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _repweight_inventory(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_parquet(path)
        repweight_cols = [
            col for col in df.columns
            if col.startswith("PWGTP") and col != "PWGTP"
        ]
        year = int(path.stem.split("_")[2])
        rows.append({
            "year": year,
            "path": _display_path(path),
            "rows": len(df),
            "repweight_columns": len(repweight_cols),
            "has_full_repweights": len(repweight_cols) == 80,
        })
    return pd.DataFrame(rows).sort_values("year")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download ACS API files with PWGTP1-PWGTP80 and audit availability.",
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_RAW,
        help="Directory where ACS raw parquet files should be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload years even if the replicate-weight parquet already exists.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Also build analysis-ready parquet files that preserve replicate weights.",
    )
    args = parser.parse_args(argv)

    downloader = ACSDownloader(raw_dir=args.output_dir)
    result = downloader.download(
        years=args.years,
        mode="api",
        include_replicate_weights=True,
        force=args.force,
    )

    if args.standardize:
        for year in args.years:
            standardize_acs_year(year, raw_variant="api_repweights")

    inventory = _repweight_inventory(result.paths)
    RESULTS_DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    inventory_path = RESULTS_DIAGNOSTICS / "acs_repweight_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    print(f"Downloaded {len(result.paths)} ACS replicate-weight parquet file(s)")
    print(f"Wrote inventory: {inventory_path}")
    if args.standardize:
        print("Repweight-aware analysis-ready parquet build requested.")


if __name__ == "__main__":
    main()
