#!/usr/bin/env python3
"""Backfill ACS API raw parquet files with ESR employment-status codes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from pathlib import Path

import httpx
import pandas as pd

from gender_gap.downloaders.acs import ACS_API_URL_TEMPLATE, ALL_STATES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "acs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("acs_esr_backfill")

DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
TARGET_SUFFIXES = {
    "api": "_api.parquet",
    "api_repweights": "_api_repweights.parquet",
}


def fetch_esr_state(year: int, state: str, api_key: str | None) -> pd.DataFrame:
    params = {"get": "SERIALNO,SPORDER,ESR", "for": f"state:{state}"}
    if api_key:
        params["key"] = api_key
    url = ACS_API_URL_TEMPLATE.format(year=year)
    response = httpx.get(url, params=params, follow_redirects=True, timeout=300)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    for col in ["SPORDER", "ESR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def download_esr_frame(year: int, max_workers: int = 6) -> pd.DataFrame:
    api_key = os.environ.get("CENSUS_API_KEY")
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_esr_state, year, state, api_key): state
            for state in ALL_STATES
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            state = futures[future]
            frames.append(future.result())
            if idx % 10 == 0 or idx == len(ALL_STATES):
                logger.info(
                    "ACS %d ESR: %d/%d states complete (last=%s)",
                    year,
                    idx,
                    len(ALL_STATES),
                    state,
                )
    return pd.concat(frames, ignore_index=True)


def backfill_target(year: int, variant: str, esr_df: pd.DataFrame, force: bool = False) -> Path | None:
    target = DATA_RAW / f"acs_pums_{year}{TARGET_SUFFIXES[variant]}"
    if not target.exists():
        logger.warning("Skipping missing target: %s", target)
        return None

    raw = pd.read_parquet(target)
    if "ESR" in raw.columns and not force:
        logger.info("Skipping already-enriched target: %s", target)
        return target

    merge_keys = [col for col in ["SERIALNO", "SPORDER", "state"] if col in raw.columns and col in esr_df.columns]
    if merge_keys[:2] != ["SERIALNO", "SPORDER"]:
        raise ValueError(f"Target missing ACS merge keys: {target}")

    raw_for_merge = raw.drop(columns=[c for c in ["ESR"] if c in raw.columns]).copy()
    esr_subset = esr_df[merge_keys + ["ESR"]].copy()
    for key in merge_keys:
        raw_for_merge[key] = raw_for_merge[key].astype(str)
        esr_subset[key] = esr_subset[key].astype(str)
    dupes = esr_subset.duplicated(subset=merge_keys, keep="last")
    if dupes.any():
        logger.warning(
            "ACS %d %s: dropping %d duplicate ESR rows before merge",
            year,
            variant,
            int(dupes.sum()),
        )
        esr_subset = esr_subset.loc[~dupes].copy()

    merged = raw_for_merge.merge(
        esr_subset,
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    merged.to_parquet(target, index=False)
    logger.info("Backfilled %s", target)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backfill existing ACS API raw parquet files with ESR.",
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(TARGET_SUFFIXES),
        default=sorted(TARGET_SUFFIXES),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args(argv)

    for year in args.years:
        esr_df = download_esr_frame(year, max_workers=args.max_workers)
        for variant in args.variants:
            backfill_target(year, variant, esr_df, force=args.force)


if __name__ == "__main__":
    main()
