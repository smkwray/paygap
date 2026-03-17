"""ACS PUMS downloader.

Supports three acquisition lanes:
1. IPUMS USA extracts (manual placeholder)
2. Official Census PUMS ZIP files
3. Census API extracts, optionally including `PWGTP1`-`PWGTP80`

The replicate-weight API lane is the preferred acquisition path for adding
survey-design uncertainty to the existing pipeline because it preserves the
current Parquet-oriented raw ingest while pulling the richer ACS weight set.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only in lean environments
    import requests as httpx
import pandas as pd

from gender_gap.downloaders.base import BaseDownloader, DownloadResult
from gender_gap.settings import DATA_RAW

logger = logging.getLogger(__name__)

# Census PUMS CSV base URL pattern (1-year person file)
ACS_PUMS_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_pus.zip"
)

ACS_API_URL_TEMPLATE = "https://api.census.gov/data/{year}/acs/acs1/pums"

ALL_STATES = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56",
]

# Census API `get=` is limited to 50 variables. Reserve two slots for the
# merge keys (`SERIALNO`, `SPORDER`) and use the remaining 48.
ACS_API_VAR_CHUNK_SIZE = 48
ACS_API_KEY_COLUMNS = ["SERIALNO", "SPORDER"]

ACS_VARS_2023PLUS = [
    "SERIALNO", "SPORDER", "SEX", "AGEP", "SCHL", "HISP", "RAC1P",
    "WAGP", "PERNP", "WKHP", "WKWN", "COW", "OCCP", "INDP", "ESR",
    "NOC", "PAOC", "FER", "MARHM", "CPLT", "PARTNER", "RELSHIPP",
    "JWTRNS", "JWMNP", "STATE", "PUMA", "PWGTP", "MAR", "ADJINC",
]

ACS_VARS_2019PLUS = [
    "SERIALNO", "SPORDER", "SEX", "AGEP", "SCHL", "HISP", "RAC1P",
    "WAGP", "PERNP", "WKHP", "WKWN", "COW", "OCCP", "INDP", "ESR",
    "NOC", "PAOC", "FER", "MARHM", "CPLT", "PARTNER", "RELSHIPP",
    "JWTRNS", "JWMNP", "ST", "PUMA", "PWGTP", "MAR", "ADJINC",
]

ACS_VARS_PRE2019 = [
    "SERIALNO", "SPORDER", "SEX", "AGEP", "SCHL", "HISP", "RAC1P",
    "WAGP", "PERNP", "WKHP", "WKW", "COW", "OCCP", "INDP", "ESR",
    "NOC", "PAOC", "FER", "MARHM", "PARTNER", "RELP",
    "JWTR", "JWMNP", "ST", "PUMA", "PWGTP", "MAR", "ADJINC",
]

ACS_NUMERIC_COLS = [
    "SEX", "AGEP", "WAGP", "PERNP", "WKHP", "WKWN", "WKW", "COW",
    "JWTRNS", "JWTR", "JWMNP", "ST", "STATE", "PWGTP", "MAR", "ADJINC", "HISP",
    "SCHL", "RAC1P", "SPORDER", "OCCP", "INDP", "ESR", "NOC", "PAOC",
    "FER", "MARHM", "CPLT", "PARTNER", "RELSHIPP", "RELP",
]
ACS_REPLICATE_WEIGHT_COLUMNS = [f"PWGTP{i}" for i in range(1, 81)]


def acs_api_variables(year: int, include_replicate_weights: bool = False) -> list[str]:
    """Return ACS API variable names for a given year."""
    if year >= 2023:
        vars_list = ACS_VARS_2023PLUS.copy()
    elif year >= 2019:
        vars_list = ACS_VARS_2019PLUS.copy()
    else:
        vars_list = ACS_VARS_PRE2019.copy()
    if include_replicate_weights:
        vars_list.extend(ACS_REPLICATE_WEIGHT_COLUMNS)
    return vars_list


def chunk_api_variables(
    variables: list[str],
    chunk_size: int = ACS_API_VAR_CHUNK_SIZE,
) -> list[list[str]]:
    """Split ACS API variables into mergeable chunks.

    Each chunk repeats the key columns required to merge responses across
    multiple API calls for the same state.
    """
    non_key = [var for var in variables if var not in ACS_API_KEY_COLUMNS]
    chunks: list[list[str]] = []
    for start in range(0, len(non_key), chunk_size):
        chunk = ACS_API_KEY_COLUMNS + non_key[start:start + chunk_size]
        chunks.append(chunk)
    return chunks


class ACSDownloader(BaseDownloader):
    """Download ACS PUMS person-level CSV files from Census."""

    dataset_id = "ACS_PUMS"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "acs")

    def download(
        self,
        years: list[int] | None = None,
        use_ipums: bool = False,
        mode: str = "official",
        include_replicate_weights: bool = False,
        force: bool = False,
        max_workers: int = 6,
        **kwargs,
    ) -> DownloadResult:
        if years is None:
            years = [2022, 2023]

        if use_ipums:
            return self._ipums_placeholder(years)

        if mode == "api":
            paths = self._download_api_years(
                years,
                include_replicate_weights=include_replicate_weights,
                force=force,
                max_workers=max_workers,
            )
            notes = (
                f"Years: {years}; mode=api; replicate_weights={include_replicate_weights}; "
                f"max_workers={max_workers}"
            )
            source_url = ACS_API_URL_TEMPLATE.format(year="<year>")
        elif mode == "official":
            paths = self._download_official_zip_years(years, force=force)
            notes = f"Years: {years}; mode=official_zip"
            source_url = ACS_PUMS_URL_TEMPLATE.format(year="<year>")
        else:
            raise ValueError(f"Unsupported ACS download mode: {mode}")

        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=paths,
            source_url=source_url,
            download_date=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )

    def _download_official_zip_years(
        self,
        years: list[int],
        force: bool = False,
    ) -> list[Path]:
        paths = []
        for year in years:
            dest = self.raw_dir / f"acs_pums_{year}.zip"
            if dest.exists() and not force:
                logger.info("ACS %d official ZIP already downloaded: %s", year, dest)
                paths.append(dest)
                continue
            url = ACS_PUMS_URL_TEMPLATE.format(year=year)
            logger.info("Downloading ACS PUMS %d official ZIP from %s", year, url)
            self._download_file(url, dest)
            paths.append(dest)
        return paths

    def _download_api_years(
        self,
        years: list[int],
        include_replicate_weights: bool = False,
        force: bool = False,
        max_workers: int = 6,
    ) -> list[Path]:
        paths = []
        for year in years:
            dest = self._api_dest_path(year, include_replicate_weights)
            if dest.exists() and not force:
                logger.info("ACS %d API parquet already downloaded: %s", year, dest)
                paths.append(dest)
                continue
            logger.info(
                "Downloading ACS PUMS %d via API (replicate_weights=%s)",
                year,
                include_replicate_weights,
            )
            df = self._download_api_year(
                year,
                include_replicate_weights=include_replicate_weights,
                max_workers=max_workers,
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dest, index=False)
            paths.append(dest)
            logger.info("ACS %d API parquet written: %s", year, dest)
        return paths

    def _api_dest_path(self, year: int, include_replicate_weights: bool) -> Path:
        suffix = "_api_repweights" if include_replicate_weights else "_api"
        return self.raw_dir / f"acs_pums_{year}{suffix}.parquet"

    def _download_api_year(
        self,
        year: int,
        include_replicate_weights: bool = False,
        max_workers: int = 6,
    ) -> pd.DataFrame:
        api_key = os.environ.get("CENSUS_API_KEY")
        variables = acs_api_variables(year, include_replicate_weights=include_replicate_weights)
        variable_chunks = chunk_api_variables(variables)
        all_states: list[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._download_state_chunks,
                    year=year,
                    state=state,
                    variable_chunks=variable_chunks,
                    api_key=api_key,
                ): state
                for state in ALL_STATES
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                state = futures[future]
                state_df = future.result()
                all_states.append(state_df)
                if idx % 10 == 0 or idx == len(ALL_STATES):
                    logger.info(
                        "ACS %d API: %d/%d states complete (last=%s, %d rows accumulated)",
                        year,
                        idx,
                        len(ALL_STATES),
                        state,
                        sum(len(frame) for frame in all_states),
                    )

        df = pd.concat(all_states, ignore_index=True)
        return self._normalize_api_frame(
            df,
            include_replicate_weights=include_replicate_weights,
        )

    def _download_state_chunks(
        self,
        year: int,
        state: str,
        variable_chunks: list[list[str]],
        api_key: str | None,
    ) -> pd.DataFrame:
        merged: pd.DataFrame | None = None
        for variables in variable_chunks:
            chunk_df = self._fetch_api_chunk(year, state, variables, api_key=api_key)
            if merged is None:
                merged = chunk_df
            else:
                merged = merged.merge(
                    chunk_df,
                    on=["SERIALNO", "SPORDER", "state"],
                    how="inner",
                    validate="one_to_one",
                )
        if merged is None:
            raise RuntimeError(f"ACS API returned no data for year={year}, state={state}")
        return merged

    def _fetch_api_chunk(
        self,
        year: int,
        state: str,
        variables: list[str],
        api_key: str | None,
    ) -> pd.DataFrame:
        params = {
            "get": ",".join(variables),
            "for": f"state:{state}",
        }
        if api_key:
            params["key"] = api_key

        url = ACS_API_URL_TEMPLATE.format(year=year)
        response = httpx.get(url, params=params, follow_redirects=True, timeout=300)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data[1:], columns=data[0])

    @staticmethod
    def _normalize_api_frame(
        df: pd.DataFrame,
        include_replicate_weights: bool = False,
    ) -> pd.DataFrame:
        numeric_cols = ACS_NUMERIC_COLS.copy()
        if include_replicate_weights:
            numeric_cols.extend(ACS_REPLICATE_WEIGHT_COLUMNS)

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "ADJINC" in df.columns and df["ADJINC"].max() < 100:
            df["ADJINC"] = df["ADJINC"] * 1_000_000

        updates: dict[str, pd.Series] = {}
        if "STATE" in df.columns and "ST" not in df.columns:
            updates["ST"] = df["STATE"]
        if "WKW" in df.columns and "WKWN" not in df.columns:
            wkw_to_weeks = {1: 50, 2: 46, 3: 39, 4: 33, 5: 20, 6: 7}
            updates["WKWN"] = df["WKW"].map(wkw_to_weeks)
        if "JWTR" in df.columns and "JWTRNS" not in df.columns:
            updates["JWTRNS"] = df["JWTR"]
        if updates:
            update_frame = pd.DataFrame(updates, index=df.index)
            df = pd.concat([df, update_frame], axis=1)
        return df

    def _ipums_placeholder(self, years: list[int]) -> DownloadResult:
        """IPUMS extracts require manual creation. Return placeholder."""
        ipums_dir = self.raw_dir / "ipums"
        ipums_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "IPUMS ACS: place your extract files in %s. "
            "Create extracts at https://usa.ipums.org/usa/",
            ipums_dir,
        )
        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[ipums_dir],
            source_url="https://usa.ipums.org/usa/",
            download_date=datetime.now(timezone.utc).isoformat(),
            notes=f"IPUMS placeholder for years {years}. Manual extract required.",
        )

    @staticmethod
    def _download_file(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 256):
                    f.write(chunk)
