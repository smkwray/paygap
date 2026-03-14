#!/usr/bin/env python3
"""Build a dominant PUMA-to-CBSA crosswalk from official Census relationship files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_puma_cbsa_crosswalk")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "crosswalks" / "geography" / "puma_to_cbsa_crosswalk.csv"

PUMA_UA_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "puma520/tab20_puma520_ua20_natl.txt"
)
UA_COUNTY_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "ua/tab20_ua20_county20_natl.txt"
)
COUNTY_CBSA_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
    "reference-files/2023/delineation-files/list1_2023.xlsx"
)


def _load_county_cbsa() -> pd.DataFrame:
    raw = pd.read_excel(COUNTY_CBSA_URL, header=2)
    raw.columns = [
        "cbsa_code",
        "metro_division_code",
        "csa_code",
        "cbsa_title",
        "area_type",
        "metro_division_title",
        "csa_title",
        "county_name",
        "state_name",
        "state_fips",
        "county_fips",
        "central_outlying",
    ]
    raw["state_fips"] = pd.to_numeric(raw["state_fips"], errors="coerce").astype("Int64").astype("string").str.zfill(2)
    raw["county_fips"] = pd.to_numeric(raw["county_fips"], errors="coerce").astype("Int64").astype("string").str.zfill(3)
    raw["county_geoid"] = raw["state_fips"] + raw["county_fips"]
    raw["cbsa_code"] = pd.to_numeric(raw["cbsa_code"], errors="coerce").astype("Int64").astype("string").str.zfill(5)
    raw["metro_status"] = raw["area_type"].map(_normalize_area_type).fillna("unknown")
    return raw[["county_geoid", "cbsa_code", "cbsa_title", "metro_status"]].dropna(subset=["county_geoid"])


def _normalize_area_type(value) -> str | None:
    if not isinstance(value, str):
        return None
    if "Metropolitan" in value:
        return "metropolitan"
    if "Micropolitan" in value:
        return "micropolitan"
    return "unknown"


def build_crosswalk() -> pd.DataFrame:
    """Build the dominant CBSA assignment for each state+PUMA pair."""
    logger.info("Loading official Census relationship files...")
    puma_ua = pd.read_csv(PUMA_UA_URL, sep="|")
    ua_county = pd.read_csv(UA_COUNTY_URL, sep="|")
    county_cbsa = _load_county_cbsa()

    logger.info("Normalizing keys...")
    puma_ua = puma_ua[puma_ua["GEOID_PUMA5_20"].notna()].copy()
    puma_ua["GEOID_PUMA5_20"] = (
        pd.to_numeric(puma_ua["GEOID_PUMA5_20"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(7)
    )
    puma_ua["state_fips"] = puma_ua["GEOID_PUMA5_20"].str[:2]
    puma_ua["puma"] = puma_ua["GEOID_PUMA5_20"].str[-5:]
    puma_ua["GEOID_UA_20"] = pd.to_numeric(puma_ua["GEOID_UA_20"], errors="coerce").astype("Int64").astype("string").str.zfill(5)
    puma_ua["AREALAND_PART"] = pd.to_numeric(puma_ua["AREALAND_PART"], errors="coerce").fillna(0)
    puma_ua["AREALAND_UA_20"] = pd.to_numeric(puma_ua["AREALAND_UA_20"], errors="coerce").replace(0, pd.NA)

    ua_county = ua_county[ua_county["GEOID_UA_20"].notna()].copy()
    ua_county["GEOID_UA_20"] = pd.to_numeric(ua_county["GEOID_UA_20"], errors="coerce").astype("Int64").astype("string").str.zfill(5)
    ua_county["county_geoid"] = pd.to_numeric(ua_county["GEOID_COUNTY_20"], errors="coerce").astype("Int64").astype("string").str.zfill(5)
    ua_county["AREALAND_PART"] = pd.to_numeric(ua_county["AREALAND_PART"], errors="coerce").fillna(0)
    ua_county["AREALAND_UA_20"] = pd.to_numeric(ua_county["AREALAND_UA_20"], errors="coerce").replace(0, pd.NA)

    logger.info("Joining PUMA->UA, UA->county, and county->CBSA...")
    merged = puma_ua[
        ["state_fips", "puma", "GEOID_UA_20", "AREALAND_PART", "AREALAND_UA_20"]
    ].merge(
        ua_county[["GEOID_UA_20", "county_geoid", "AREALAND_PART"]],
        how="left",
        on="GEOID_UA_20",
        suffixes=("_puma_ua", "_ua_county"),
    )
    merged = merged.merge(county_cbsa, how="left", on="county_geoid")

    # Approximate overlap by allocating each PUMA-UA overlap across counties in proportion
    # to the county's share of the same urban area's land area.
    merged["estimated_overlap_area"] = (
        merged["AREALAND_PART_puma_ua"] * merged["AREALAND_PART_ua_county"] / merged["AREALAND_UA_20"]
    )
    merged["estimated_overlap_area"] = pd.to_numeric(
        merged["estimated_overlap_area"], errors="coerce"
    ).fillna(0)

    scored = (
        merged.dropna(subset=["cbsa_code"])
        .groupby(["state_fips", "puma", "cbsa_code", "cbsa_title", "metro_status"], as_index=False)["estimated_overlap_area"]
        .sum()
    )

    dominant = (
        scored.sort_values(
            ["state_fips", "puma", "estimated_overlap_area", "cbsa_code"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["state_fips", "puma"])
        .rename(columns={"estimated_overlap_area": "dominant_overlap_area"})
    )

    all_pumas = puma_ua[["state_fips", "puma"]].drop_duplicates()
    result = all_pumas.merge(dominant, how="left", on=["state_fips", "puma"])
    result["metro_status"] = result["metro_status"].fillna("noncore")

    output = result[
        ["state_fips", "puma", "cbsa_code", "cbsa_title", "metro_status", "dominant_overlap_area"]
    ].sort_values(["state_fips", "puma"])
    return output


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    crosswalk = build_crosswalk()
    crosswalk.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote %d PUMA rows to %s", len(crosswalk), OUTPUT_PATH)


if __name__ == "__main__":
    main()
