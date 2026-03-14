#!/usr/bin/env python3
"""Download ALL datasets for the gender gap project.

Downloads:
  1. ACS PUMS 2015-2023 via Census API (9 years × 51 states)
  2. CPS ASEC 2015-2023 via Census API (March supplement with earnings)
  3. ATUS 2003-2023 from BLS (activity summary + respondent + CPS link)
  4. Context: LAUS, CPI-U, QCEW, OEWS from BLS
  5. BEA Regional Price Parities via BEA API

All data cached to data/raw/<source>/ as Parquet files.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

handler = FlushHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("download_all")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# API keys (from .env or shell environment)
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
BEA_API_KEY = os.getenv("BEA_API_KEY", "")
BLS_API_KEY = os.getenv("BLS_API_KEY", "")

# All 50 states + DC
ALL_STATES = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56",
]

import threading

_thread_local = threading.local()

def get_session() -> requests.Session:
    """Thread-local requests session."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update({"User-Agent": "GenderGapResearch/1.0"})
    return _thread_local.session


# ═══════════════════════════════════════════════════════════════════
# 1. ACS PUMS (Census API)
# ═══════════════════════════════════════════════════════════════════

# 2023+ variable names (ST -> STATE)
ACS_VARS_2023PLUS = [
    "SERIALNO", "SPORDER", "SEX", "AGEP", "SCHL", "HISP", "RAC1P",
    "WAGP", "PERNP", "WKHP", "WKWN", "COW", "OCCP", "INDP",
    "JWTRNS", "JWMNP", "STATE", "PUMA", "PWGTP", "MAR", "ADJINC",
]

# 2019-2022 variable names
ACS_VARS_2019PLUS = [
    "SERIALNO", "SPORDER", "SEX", "AGEP", "SCHL", "HISP", "RAC1P",
    "WAGP", "PERNP", "WKHP", "WKWN", "COW", "OCCP", "INDP",
    "JWTRNS", "JWMNP", "ST", "PUMA", "PWGTP", "MAR", "ADJINC",
]

# Pre-2019: WKW instead of WKWN, JWTR instead of JWTRNS
ACS_VARS_PRE2019 = [
    "SERIALNO", "SPORDER", "SEX", "AGEP", "SCHL", "HISP", "RAC1P",
    "WAGP", "PERNP", "WKHP", "WKW", "COW", "OCCP", "INDP",
    "JWTR", "JWMNP", "ST", "PUMA", "PWGTP", "MAR", "ADJINC",
]

ACS_NUMERIC_COLS = [
    "SEX", "AGEP", "WAGP", "PERNP", "WKHP", "WKWN", "WKW", "COW",
    "JWTRNS", "JWTR", "JWMNP", "ST", "STATE", "PWGTP", "MAR", "ADJINC", "HISP",
    "SCHL", "RAC1P", "SPORDER", "OCCP", "INDP",
]


def download_acs_year(year: int) -> pd.DataFrame:
    """Download ACS PUMS for a single year via Census API."""
    cache_path = DATA_RAW / "acs" / f"acs_pums_{year}_api.parquet"
    if cache_path.exists():
        logger.info("ACS %d: cached (%s)", year, cache_path)
        return pd.read_parquet(cache_path)

    logger.info("ACS %d: downloading via Census API...", year)
    # Variable names changed in 2019: WKWN/JWTRNS replaced WKW/JWTR
    # In 2023: ST renamed to STATE
    if year >= 2023:
        vars_list = ACS_VARS_2023PLUS
    elif year >= 2019:
        vars_list = ACS_VARS_2019PLUS
    else:
        vars_list = ACS_VARS_PRE2019

    vars_str = ",".join(vars_list)
    all_rows = []
    header = None

    for i, st in enumerate(ALL_STATES):
        url = (
            f"https://api.census.gov/data/{year}/acs/acs1/pums"
            f"?get={vars_str}&for=state:{st}&key={CENSUS_API_KEY}"
        )
        for attempt in range(3):
            try:
                r = get_session().get(url, timeout=120)
                r.raise_for_status()
                data = r.json()
                if header is None:
                    header = data[0]
                all_rows.extend(data[1:])
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning("ACS %d state %s attempt %d: %s", year, st, attempt + 1, e)
                    time.sleep(3 * (attempt + 1))
                else:
                    logger.error("ACS %d state %s FAILED: %s", year, st, e)

        if (i + 1) % 10 == 0:
            logger.info("  ACS %d: %d/%d states, %d rows", year, i + 1, len(ALL_STATES), len(all_rows))

    logger.info("ACS %d: %d total records", year, len(all_rows))
    df = pd.DataFrame(all_rows, columns=header)

    # Convert numeric columns
    for col in ACS_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix ADJINC (API returns decimal, standardizer expects integer * 1M)
    if "ADJINC" in df.columns and df["ADJINC"].max() < 100:
        df["ADJINC"] = df["ADJINC"] * 1_000_000

    # Normalize variable names across years to 2019-2022 conventions
    # For 2023+: STATE -> ST
    if "STATE" in df.columns and "ST" not in df.columns:
        df["ST"] = df["STATE"]
    # For pre-2019: WKW -> WKWN, JWTR -> JWTRNS
    if "WKW" in df.columns and "WKWN" not in df.columns:
        wkw_to_weeks = {1: 50, 2: 46, 3: 39, 4: 33, 5: 20, 6: 7}
        df["WKWN"] = df["WKW"].map(wkw_to_weeks)
    if "JWTR" in df.columns and "JWTRNS" not in df.columns:
        df["JWTRNS"] = df["JWTR"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("ACS %d: cached to %s", year, cache_path)
    return df


def download_all_acs(years: list[int]) -> dict[int, pd.DataFrame]:
    """Download ACS PUMS for multiple years."""
    results = {}
    for year in years:
        try:
            results[year] = download_acs_year(year)
        except Exception as e:
            logger.error("ACS %d failed: %s", year, e)
    return results


# ═══════════════════════════════════════════════════════════════════
# 2. CPS ASEC (Census API - March Annual Social & Economic Supplement)
# ═══════════════════════════════════════════════════════════════════

CPS_ASEC_VARS = [
    "A_SEX", "A_AGE", "A_HGA", "PRDTRACE", "PEHSPNON",
    "A_MJOCC", "A_MJIND", "A_CLSWKR", "A_USLHRS", "A_HRS1",
    "ERN_VAL", "WSAL_VAL", "WS_VAL", "A_WKSTAT",
    "A_MARITL", "FOWNU6", "FOWNU18",
    "GESTFIPS", "MARSUPWT",
]

CPS_NUMERIC_COLS = [
    "A_SEX", "A_AGE", "A_HGA", "PRDTRACE", "PEHSPNON",
    "A_MJOCC", "A_MJIND", "A_CLSWKR", "A_USLHRS", "A_HRS1",
    "ERN_VAL", "WSAL_VAL", "WS_VAL", "A_WKSTAT",
    "A_MARITL", "FOWNU6", "FOWNU18",
    "GESTFIPS", "MARSUPWT",
]


def download_cps_asec_year(year: int) -> pd.DataFrame:
    """Download CPS ASEC for a single year via Census API."""
    cache_path = DATA_RAW / "cps" / f"cps_asec_{year}_api.parquet"
    if cache_path.exists():
        logger.info("CPS ASEC %d: cached (%s)", year, cache_path)
        return pd.read_parquet(cache_path)

    logger.info("CPS ASEC %d: downloading via Census API...", year)

    # Try different API endpoint patterns
    # CPS ASEC is at /cps/asec/mar for most years
    vars_str = ",".join(CPS_ASEC_VARS)
    all_rows = []
    header = None

    # CPS ASEC API doesn't require state-by-state; get all at once
    # But may need to paginate or query by state for large datasets
    for st in ALL_STATES:
        url = (
            f"https://api.census.gov/data/{year}/cps/asec/mar"
            f"?get={vars_str}&for=state:{st}&key={CENSUS_API_KEY}"
        )
        for attempt in range(3):
            try:
                r = get_session().get(url, timeout=120)
                r.raise_for_status()
                data = r.json()
                if header is None:
                    header = data[0]
                all_rows.extend(data[1:])
                break
            except requests.exceptions.HTTPError as e:
                if r.status_code == 404:
                    # Try alternative endpoint
                    alt_url = (
                        f"https://api.census.gov/data/{year}/cps/asec/person"
                        f"?get={vars_str}&for=state:{st}&key={CENSUS_API_KEY}"
                    )
                    try:
                        r2 = get_session().get(alt_url, timeout=120)
                        r2.raise_for_status()
                        data = r2.json()
                        if header is None:
                            header = data[0]
                        all_rows.extend(data[1:])
                        break
                    except Exception:
                        pass
                if attempt < 2:
                    logger.warning("CPS %d state %s attempt %d: %s", year, st, attempt + 1, e)
                    time.sleep(3)
                else:
                    logger.error("CPS %d state %s FAILED: %s", year, st, e)
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    logger.error("CPS %d state %s FAILED: %s", year, st, e)

    if not all_rows:
        logger.error("CPS ASEC %d: no data retrieved", year)
        return pd.DataFrame()

    logger.info("CPS ASEC %d: %d total records", year, len(all_rows))
    df = pd.DataFrame(all_rows, columns=header)

    for col in CPS_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("CPS ASEC %d: cached to %s", year, cache_path)
    return df


def download_all_cps(years: list[int]) -> dict[int, pd.DataFrame]:
    """Download CPS ASEC for multiple years."""
    results = {}
    for year in years:
        try:
            results[year] = download_cps_asec_year(year)
        except Exception as e:
            logger.error("CPS ASEC %d failed: %s", year, e)
    return results


# ═══════════════════════════════════════════════════════════════════
# 3. ATUS (BLS Direct Downloads)
# ═══════════════════════════════════════════════════════════════════

ATUS_FILES = {
    "respondent": "https://www.bls.gov/tus/datafiles/atusresp-0323.zip",
    "activity_summary": "https://www.bls.gov/tus/datafiles/atussum-0323.zip",
    "roster": "https://www.bls.gov/tus/datafiles/atusrost-0323.zip",
    "cps_link": "https://www.bls.gov/tus/datafiles/atuscps-0323.zip",
}


def download_atus() -> dict[str, pd.DataFrame]:
    """Download ATUS data files from BLS."""
    atus_dir = DATA_RAW / "atus"
    atus_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, url in ATUS_FILES.items():
        cache_path = atus_dir / f"atus_{name}.parquet"
        if cache_path.exists():
            logger.info("ATUS %s: cached", name)
            results[name] = pd.read_parquet(cache_path)
            continue

        logger.info("ATUS %s: downloading from %s", name, url)
        try:
            r = get_session().get(url, timeout=300)
            r.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                # Find the .dat or .csv file inside
                csv_files = [f for f in zf.namelist()
                             if f.endswith(('.dat', '.csv')) and not f.startswith('__')]
                if not csv_files:
                    logger.warning("ATUS %s: no data file in zip", name)
                    continue

                with zf.open(csv_files[0]) as f:
                    df = pd.read_csv(f)

            df.to_parquet(cache_path, index=False)
            results[name] = df
            logger.info("ATUS %s: %d rows, cached", name, len(df))
        except Exception as e:
            logger.error("ATUS %s failed: %s", name, e)

    return results


# ═══════════════════════════════════════════════════════════════════
# 4. CONTEXT DATA (BLS Direct Downloads)
# ═══════════════════════════════════════════════════════════════════

def download_laus() -> pd.DataFrame:
    """Download Local Area Unemployment Statistics from BLS."""
    cache_path = DATA_RAW / "context" / "laus_states.parquet"
    if cache_path.exists():
        logger.info("LAUS: cached")
        return pd.read_parquet(cache_path)

    logger.info("LAUS: downloading state-level data...")
    url = "https://download.bls.gov/pub/time.series/la/la.data.3.AllStatesS"
    try:
        r = get_session().get(url, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="\t")
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        df["value"] = pd.to_numeric(df["value"].astype(str).str.strip(), errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("LAUS: %d rows cached", len(df))
        return df
    except Exception as e:
        logger.error("LAUS failed: %s", e)
        return pd.DataFrame()


def download_cpi_u() -> pd.DataFrame:
    """Download CPI-U annual averages from BLS."""
    cache_path = DATA_RAW / "context" / "cpi_u.parquet"
    if cache_path.exists():
        logger.info("CPI-U: cached")
        return pd.read_parquet(cache_path)

    logger.info("CPI-U: downloading...")
    url = "https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems"
    try:
        r = get_session().get(url, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="\t")
        df.columns = [c.strip() for c in df.columns]
        df["value"] = pd.to_numeric(df["value"].astype(str).str.strip(), errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # Filter to annual average (M13) for all urban consumers
        annual = df[
            (df["series_id"].str.strip() == "CUUR0000SA0") &
            (df["period"].str.strip() == "M13")
        ].copy()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        annual.to_parquet(cache_path, index=False)
        logger.info("CPI-U: %d annual observations cached", len(annual))
        return annual
    except Exception as e:
        logger.error("CPI-U failed: %s", e)
        return pd.DataFrame()


def download_qcew(years: list[int]) -> dict[int, pd.DataFrame]:
    """Download QCEW annual data from BLS."""
    qcew_dir = DATA_RAW / "context" / "qcew"
    qcew_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for year in years:
        cache_path = qcew_dir / f"qcew_{year}.parquet"
        if cache_path.exists():
            logger.info("QCEW %d: cached", year)
            results[year] = pd.read_parquet(cache_path)
            continue

        url = f"https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip"
        logger.info("QCEW %d: downloading...", year)
        try:
            r = get_session().get(url, timeout=300)
            r.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                if csv_files:
                    with zf.open(csv_files[0]) as f:
                        df = pd.read_csv(
                            f,
                            dtype={
                                "area_fips": "string",
                                "industry_code": "string",
                                "agglvl_code": "string",
                                "own_code": "Int64",
                            },
                            low_memory=False,
                        )
                    df["area_fips"] = (
                        df["area_fips"].astype("string").str.replace(r"\\.0$", "", regex=True).str.zfill(5)
                    )
                    # Filter to total private (own_code=5) and total all (own_code=0)
                    df_filtered = df[df["own_code"].isin([0, 5])].copy()
                    df_filtered.to_parquet(cache_path, index=False)
                    results[year] = df_filtered
                    logger.info("QCEW %d: %d rows cached", year, len(df_filtered))
        except Exception as e:
            logger.error("QCEW %d failed: %s", year, e)

    return results


def download_oews(years: list[int]) -> dict[int, pd.DataFrame]:
    """Download or stage OEWS (Occupational Employment & Wage Statistics) data.

    BLS currently blocks direct programmatic access from this environment, so this
    function first looks for manually staged ZIP/XLSX/CSV files in the raw
    directory. If none are present, it writes DOWNLOAD_INSTRUCTIONS.md and returns
    an empty result.
    """
    oews_dir = DATA_RAW / "context" / "oews"
    oews_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    instructions = oews_dir / "DOWNLOAD_INSTRUCTIONS.md"
    if not instructions.exists():
        instructions.write_text(
            "# OEWS Manual Staging\n\n"
            "BLS OEWS files are returning access-denied responses from this environment.\n"
            "Download the needed archives from the official BLS OEWS tables page in a browser:\n"
            "https://www.bls.gov/oes/tables.htm\n\n"
            "Typical direct archive patterns listed on that page include:\n"
            "- https://www.bls.gov/oes/special-requests/oesm23st.zip\n"
            "- https://www.bls.gov/oes/special-requests/oesm23all.zip\n\n"
            "Place the ZIP/XLSX/CSV files in this directory and rerun the context build.\n"
        )

    for year in years:
        cache_path = oews_dir / f"oews_{year}.parquet"
        if cache_path.exists():
            logger.info("OEWS %d: cached", year)
            results[year] = pd.read_parquet(cache_path)
            continue

        yy = str(year)[-2:]
        staged = sorted(oews_dir.glob(f"*{yy}*.zip")) + sorted(oews_dir.glob(f"*{yy}*.xlsx")) + sorted(oews_dir.glob(f"*{yy}*.csv"))
        if not staged:
            logger.warning("OEWS %d: no staged local files found; see %s", year, instructions)
            continue

        try:
            src = staged[0]
            logger.info("OEWS %d: parsing staged file %s", year, src.name)
            if src.suffix == ".zip":
                with zipfile.ZipFile(src) as zf:
                    data_files = [
                        name for name in zf.namelist()
                        if name.lower().endswith((".xlsx", ".xls", ".csv"))
                    ]
                    if not data_files:
                        raise ValueError(f"No data file found inside {src.name}")
                    with zf.open(data_files[0]) as f:
                        if data_files[0].lower().endswith((".xlsx", ".xls")):
                            df = pd.read_excel(f)
                        else:
                            df = pd.read_csv(f, low_memory=False)
            elif src.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(src)
            else:
                df = pd.read_csv(src, low_memory=False)

            df.to_parquet(cache_path, index=False)
            results[year] = df
            logger.info("OEWS %d: %d rows cached", year, len(df))
        except Exception as e:
            logger.error("OEWS %d failed: %s", year, e)

    return results


# ═══════════════════════════════════════════════════════════════════
# 5. BEA REGIONAL PRICE PARITIES (BEA API)
# ═══════════════════════════════════════════════════════════════════

def download_bea_rpp() -> pd.DataFrame:
    """Download Regional Price Parities via BEA API."""
    cache_path = DATA_RAW / "context" / "bea_rpp.parquet"
    if cache_path.exists():
        logger.info("BEA RPP: cached")
        return pd.read_parquet(cache_path)

    logger.info("BEA RPP: downloading via API...")
    # BEA API for Regional Price Parities (SARPP table)
    all_data = []

    for line_code in ["1"]:  # 1 = All items RPP
        url = (
            f"https://apps.bea.gov/api/data/"
            f"?UserID={BEA_API_KEY}"
            f"&method=GetData"
            f"&DatasetName=Regional"
            f"&TableName=SARPP"
            f"&GeoFips=STATE"
            f"&LineCode={line_code}"
            f"&Year=ALL"
            f"&ResultFormat=JSON"
        )
        try:
            r = get_session().get(url, timeout=120)
            r.raise_for_status()
            resp = r.json()

            if "BEAAPI" in resp and "Results" in resp["BEAAPI"]:
                data = resp["BEAAPI"]["Results"].get("Data", [])
                if data:
                    all_data.extend(data)
                    logger.info("BEA RPP: %d observations", len(data))
        except Exception as e:
            logger.error("BEA RPP line %s failed: %s", line_code, e)

    if not all_data:
        logger.error("BEA RPP: no data retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # Standardize columns
    if "TimePeriod" in df.columns:
        df["year"] = pd.to_numeric(df["TimePeriod"], errors="coerce")
    if "DataValue" in df.columns:
        df["rpp"] = pd.to_numeric(df["DataValue"], errors="coerce")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("BEA RPP: %d rows cached", len(df))
    return df


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    start = time.time()
    # ACS 2020 1-year was not released (COVID collection suspension)
    ACS_YEARS = [y for y in range(2015, 2024) if y != 2020]
    CPS_YEARS = list(range(2015, 2024))
    QCEW_YEARS = list(range(2018, 2024))
    OEWS_YEARS = list(range(2019, 2024))

    logger.info("=" * 75)
    logger.info("DOWNLOADING ALL DATASETS (PARALLEL)")
    logger.info("=" * 75)

    # Launch all download streams in parallel:
    # - ACS years: 2 concurrent (Census API rate limit)
    # - CPS years: 2 concurrent
    # - ATUS, LAUS, CPI-U, BEA RPP: 1 each (fast)
    # - QCEW, OEWS: 1 each (BLS rate limit)

    acs_results = {}
    cps_results = {}
    other_results = {}

    def _download_acs_year(year):
        """Wrapper for thread pool."""
        try:
            df = download_acs_year(year)
            return ("acs", year, df)
        except Exception as e:
            logger.error("ACS %d failed: %s", year, e)
            return ("acs", year, None)

    def _download_cps_year(year):
        try:
            df = download_cps_asec_year(year)
            return ("cps", year, df)
        except Exception as e:
            logger.error("CPS %d failed: %s", year, e)
            return ("cps", year, None)

    def _download_context():
        """Download all context sources (LAUS, CPI-U, QCEW, OEWS, BEA RPP)."""
        results = {}
        results["laus"] = download_laus()
        logger.info("LAUS: %d rows", len(results["laus"]))

        results["cpi"] = download_cpi_u()
        logger.info("CPI-U: %d rows", len(results["cpi"]))

        results["qcew"] = download_qcew(QCEW_YEARS)
        logger.info("QCEW: %d years", len(results["qcew"]))

        results["oews"] = download_oews(OEWS_YEARS)
        logger.info("OEWS: %d years", len(results["oews"]))

        results["bea_rpp"] = download_bea_rpp()
        logger.info("BEA RPP: %d rows", len(results["bea_rpp"]))

        return ("context", 0, results)

    def _download_atus():
        try:
            data = download_atus()
            return ("atus", 0, data)
        except Exception as e:
            logger.error("ATUS failed: %s", e)
            return ("atus", 0, None)

    # Use ThreadPoolExecutor with limited workers to control Census API load
    # 4 workers: 2 for ACS years, 1 for CPS, 1 for ATUS/context
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []

        # Submit ACS years
        for year in ACS_YEARS:
            futures.append(pool.submit(_download_acs_year, year))

        # Submit CPS years
        for year in CPS_YEARS:
            futures.append(pool.submit(_download_cps_year, year))

        # Submit ATUS
        futures.append(pool.submit(_download_atus))

        # Submit context data
        futures.append(pool.submit(_download_context))

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                source, year, data = future.result()
                if source == "acs" and data is not None:
                    acs_results[year] = data
                    logger.info("✓ ACS %d: %d rows", year, len(data))
                elif source == "cps" and data is not None and len(data) > 0:
                    cps_results[year] = data
                    logger.info("✓ CPS %d: %d rows", year, len(data))
                elif source == "atus":
                    other_results["atus"] = data
                    logger.info("✓ ATUS: %d files", len(data) if data else 0)
                elif source == "context":
                    other_results["context"] = data
                    logger.info("✓ Context data downloaded")
            except Exception as e:
                logger.error("Future failed: %s", e)

    # ── Summary ──
    elapsed = time.time() - start
    logger.info("\n" + "=" * 75)
    logger.info("DOWNLOAD COMPLETE (%.1f minutes)", elapsed / 60)
    logger.info("=" * 75)

    logger.info("ACS: %d/%d years", len(acs_results), len(ACS_YEARS))
    for year in sorted(acs_results):
        logger.info("  ACS %d: %d rows", year, len(acs_results[year]))

    logger.info("CPS: %d/%d years", len(cps_results), len(CPS_YEARS))
    for year in sorted(cps_results):
        logger.info("  CPS %d: %d rows", year, len(cps_results[year]))

    # Print data inventory
    total_bytes = 0
    for path in DATA_RAW.rglob("*.parquet"):
        size = path.stat().st_size
        total_bytes += size
    logger.info("Total raw data: %.1f MB in %d files",
                total_bytes / 1e6,
                len(list(DATA_RAW.rglob("*.parquet"))))


if __name__ == "__main__":
    main()
