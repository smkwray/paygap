#!/usr/bin/env python3
"""Backfill ACS API raw parquet files with person-level family summaries.

This fetches only the ACS PUMS `NOC` and `PAOC` variables from the Census API
and merges them onto existing raw API parquet files. It avoids rewriting the
processed analysis-ready parquet files, which is useful when downstream jobs
are actively reading them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import sys
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "acs"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gender_gap.downloaders.acs import ACS_API_URL_TEMPLATE, ALL_STATES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("acs_family_backfill")

DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
FAMILY_VARS = ["SERIALNO", "SPORDER", "NOC", "PAOC"]
TARGET_SUFFIXES = {
    "api": "_api.parquet",
    "api_repweights": "_api_repweights.parquet",
}


def fetch_family_state(year: int, state: str, api_key: str | None) -> pd.DataFrame:
    params = {
        "get": ",".join(FAMILY_VARS),
        "for": f"state:{state}",
    }
    if api_key:
        params["key"] = api_key

    url = ACS_API_URL_TEMPLATE.format(year=year)
    response = httpx.get(url, params=params, follow_redirects=True, timeout=300)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    for col in ["SPORDER", "NOC", "PAOC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def download_family_frame(year: int, max_workers: int = 6) -> pd.DataFrame:
    api_key = os.environ.get("CENSUS_API_KEY")
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_family_state, year, state, api_key): state
            for state in ALL_STATES
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            state = futures[future]
            frames.append(future.result())
            if idx % 10 == 0 or idx == len(ALL_STATES):
                logger.info(
                    "ACS %d family vars: %d/%d states complete (last=%s)",
                    year,
                    idx,
                    len(ALL_STATES),
                    state,
                )
    return pd.concat(frames, ignore_index=True)


def backfill_target(
    year: int,
    variant: str,
    family_df: pd.DataFrame,
    force: bool = False,
) -> Path | None:
    target = DATA_RAW / f"acs_pums_{year}{TARGET_SUFFIXES[variant]}"
    if not target.exists():
        logger.warning("Skipping missing target: %s", target)
        return None

    raw = pd.read_parquet(target)
    if {"NOC", "PAOC"}.issubset(raw.columns) and not force:
        logger.info("Skipping already-enriched target: %s", target)
        return target

    merge_keys = [col for col in ["SERIALNO", "SPORDER", "state"] if col in raw.columns and col in family_df.columns]
    if merge_keys[:2] != ["SERIALNO", "SPORDER"]:
        raise ValueError(f"Target missing ACS merge keys: {target}")

    raw_for_merge = raw.drop(columns=[c for c in ["NOC", "PAOC"] if c in raw.columns]).copy()
    family_subset = family_df[merge_keys + ["NOC", "PAOC"]].copy()
    for key in merge_keys:
        raw_for_merge[key] = raw_for_merge[key].astype(str)
        family_subset[key] = family_subset[key].astype(str)
    dupes = family_subset.duplicated(subset=merge_keys, keep="last")
    if dupes.any():
        logger.warning(
            "ACS %d %s: dropping %d duplicate family rows before merge",
            year,
            variant,
            int(dupes.sum()),
        )
        family_subset = family_subset.loc[~dupes].copy()

    merged = raw_for_merge.merge(
        family_subset,
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    merged.to_parquet(target, index=False)
    logger.info("Backfilled %s", target)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backfill existing ACS API raw parquet files with NOC and PAOC.",
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(TARGET_SUFFIXES),
        default=sorted(TARGET_SUFFIXES),
        help="Raw ACS variants to enrich in place.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite targets even if they already contain NOC/PAOC.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Parallel state requests per year.",
    )
    args = parser.parse_args(argv)

    for year in args.years:
        family_df = download_family_frame(year, max_workers=args.max_workers)
        for variant in args.variants:
            backfill_target(year, variant, family_df, force=args.force)


if __name__ == "__main__":
    main()
