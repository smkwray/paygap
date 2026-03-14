"""Contextual data downloaders: LAUS, QCEW, QWI, O*NET, CPI-U, OEWS, BEA RPP."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from gender_gap.downloaders.base import BaseDownloader, DownloadResult
from gender_gap.settings import DATA_RAW

logger = logging.getLogger(__name__)

QCEW_OPEN_DATA_URL = "https://www.bls.gov/cew/additional-resources/open-data/"
OEWS_TABLES_URL = "https://www.bls.gov/oes/tables.htm"
OEWS_STATE_ARCHIVE_TEMPLATE = "https://www.bls.gov/oes/special-requests/oesm{yy}st.zip"
OEWS_ALL_AREAS_ARCHIVE_TEMPLATE = "https://www.bls.gov/oes/special-requests/oesm{yy}all.zip"


def _write_oews_instructions(raw_dir: Path, years: list[int]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "DOWNLOAD_INSTRUCTIONS.md"

    year_lines = []
    for year in years:
        yy = str(year)[-2:]
        year_lines.append(
            f"- {year}: state archive `{OEWS_STATE_ARCHIVE_TEMPLATE.format(yy=yy)}` "
            f"or all-areas archive `{OEWS_ALL_AREAS_ARCHIVE_TEMPLATE.format(yy=yy)}`"
        )

    path.write_text(
        "# OEWS Manual Staging\n\n"
        "BLS OEWS files are currently returning access-denied responses from this environment.\n"
        "Use the official BLS OEWS tables page to fetch the archives in a browser, then place "
        "the downloaded ZIP/XLSX/CSV files in this directory.\n\n"
        "## Official source\n"
        f"- OEWS tables: {OEWS_TABLES_URL}\n\n"
        "## Typical archive patterns\n"
        + "\n".join(year_lines)
        + "\n\n"
        "## Use in this repo\n"
        "- Keep these files under `data/raw/context/oews/`\n"
        "- If a ZIP contains multiple files, prefer the state/all-areas workbook or CSV\n"
        "- After staging files locally, rerun the context status/build scripts\n"
    )
    return path


class LAUSDownloader(BaseDownloader):
    """Local Area Unemployment Statistics from BLS."""

    dataset_id = "LAUS"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "context" / "laus")

    def download(self, years: list[int] | None = None, **kwargs) -> DownloadResult:
        # BLS flat files at https://download.bls.gov/pub/time.series/la/
        files_to_get = {
            "la.data.64.County": "la.data.64.County",
            "la.data.60.Metro": "la.data.60.Metro",
            "la.data.3.AllStatesS": "la.data.3.AllStatesS",
        }
        base_url = "https://download.bls.gov/pub/time.series/la/"
        paths = []
        for fname, dest_name in files_to_get.items():
            dest = self.raw_dir / dest_name
            if dest.exists():
                logger.info("LAUS %s already present", dest_name)
                paths.append(dest)
                continue
            url = base_url + fname
            logger.info("Downloading LAUS %s", url)
            self._download_text(url, dest)
            paths.append(dest)

        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=paths,
            source_url=base_url,
            download_date=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _download_text(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(url, follow_redirects=True, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)


class CPIDownloader(BaseDownloader):
    """CPI-U annual averages from BLS."""

    dataset_id = "CPI_U"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "context" / "cpi")

    def download(self, years: list[int] | None = None, **kwargs) -> DownloadResult:
        url = "https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems"
        dest = self.raw_dir / "cu.data.1.AllItems"
        if not dest.exists():
            logger.info("Downloading CPI-U data")
            resp = httpx.get(url, follow_redirects=True, timeout=120)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)

        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[dest],
            source_url=url,
            download_date=datetime.now(timezone.utc).isoformat(),
        )


class ONETDownloader(BaseDownloader):
    """O*NET database download."""

    dataset_id = "ONET"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "context" / "onet")

    def download(self, years: list[int] | None = None, **kwargs) -> DownloadResult:
        logger.info(
            "O*NET: download database files from "
            "https://www.onetcenter.org/database.html#all-files "
            "and place in %s",
            self.raw_dir,
        )
        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[self.raw_dir],
            source_url="https://www.onetcenter.org/database.html#all-files",
            download_date=datetime.now(timezone.utc).isoformat(),
            notes="Manual download required. Place extracted files in raw_dir.",
        )


class QCEWDownloader(BaseDownloader):
    """Quarterly Census of Employment and Wages from BLS."""

    dataset_id = "QCEW"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "context" / "qcew")

    def download(self, years: list[int] | None = None, **kwargs) -> DownloadResult:
        """Download QCEW annual average CSV slices from BLS open data.

        URL pattern from https://www.bls.gov/cew/additional-resources/open-data/
        """
        if years is None:
            years = [2022, 2023]

        base_url = "https://data.bls.gov/cew/data/files"
        paths = []
        for year in years:
            url = f"{base_url}/{year}/csv/{year}_annual_singlefile.zip"
            dest = self.raw_dir / f"{year}_annual_singlefile.zip"
            if dest.exists():
                logger.info("QCEW %d already present", year)
                paths.append(dest)
                continue
            logger.info("Downloading QCEW %d from %s", year, url)
            try:
                resp = httpx.get(url, follow_redirects=True, timeout=300)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                paths.append(dest)
                self._log_download(url, dest)
            except httpx.HTTPError as e:
                logger.warning("Failed to download QCEW %d: %s", year, e)

        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=paths,
            source_url="https://www.bls.gov/cew/additional-resources/open-data/",
            download_date=datetime.now(timezone.utc).isoformat(),
        )


class QWIDownloader(BaseDownloader):
    """Quarterly Workforce Indicators from Census LEHD."""

    dataset_id = "QWI"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "context" / "qwi")

    def download(self, years: list[int] | None = None, **kwargs) -> DownloadResult:
        """Stage QWI data via Census API.

        The QWI API at https://api.census.gov/data/timeseries/qwi/ requires
        state-by-state queries. This creates the directory and instructions.
        """
        instructions = self.raw_dir / "DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            "# QWI Data Download\n\n"
            "QWI data can be accessed via the Census API:\n"
            "  https://api.census.gov/data/timeseries/qwi/\n\n"
            "Or via the QWI Explorer:\n"
            "  https://qwiexplorer.ces.census.gov/\n\n"
            "## Recommended query parameters\n"
            "- Indicators: EarnS, EmpS\n"
            "- Dimensions: sex, education, industry, firmsize, firmage\n"
            "- Geography: state, county\n"
            "- Time: quarterly for desired years\n\n"
            "Place downloaded CSV files in this directory.\n"
        )
        logger.info(
            "QWI: query Census API or use QWI Explorer. "
            "See instructions in %s", self.raw_dir,
        )
        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[self.raw_dir],
            source_url="https://api.census.gov/data/timeseries/qwi/",
            download_date=datetime.now(timezone.utc).isoformat(),
            notes="QWI requires state-by-state API queries. Instructions written.",
        )


class OEWSDownloader(BaseDownloader):
    """Occupational Employment and Wage Statistics from BLS."""

    dataset_id = "OEWS"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "context" / "oews")

    def download(self, years: list[int] | None = None, **kwargs) -> DownloadResult:
        """Download OEWS national and state data from BLS.

        URL pattern: https://www.bls.gov/oes/special-requests/oesm{YY}nat.zip
        """
        if years is None:
            years = [2023]

        paths = []
        instructions = _write_oews_instructions(self.raw_dir, years)
        for year in years:
            yy = str(year)[-2:]
            url = OEWS_STATE_ARCHIVE_TEMPLATE.format(yy=yy)
            dest = self.raw_dir / f"oesm{yy}st.zip"
            if dest.exists():
                logger.info("OEWS %d already present", year)
                paths.append(dest)
                continue
            logger.info("Downloading OEWS %d from %s", year, url)
            try:
                resp = httpx.get(url, follow_redirects=True, timeout=300)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                paths.append(dest)
                self._log_download(url, dest)
            except httpx.HTTPError as e:
                logger.warning(
                    "Failed to download OEWS %d: %s. "
                    "Manual download from %s",
                    year, e,
                    OEWS_TABLES_URL,
                )

        if instructions.exists():
            paths.append(instructions)

        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=paths,
            source_url=OEWS_TABLES_URL,
            download_date=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Automatic OEWS download may be blocked by BLS access controls from this "
                "environment. See DOWNLOAD_INSTRUCTIONS.md for the manual staging path."
            ),
        )


class BEARPPDownloader(BaseDownloader):
    """Regional Price Parities from BEA."""

    dataset_id = "BEA_RPP"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "context" / "bea_rpp")

    def download(self, years: list[int] | None = None, **kwargs) -> DownloadResult:
        """Stage BEA RPP data.

        BEA RPP tables require manual download from:
        https://www.bea.gov/data/prices-inflation/regional-price-parities-state-and-metro-area
        """
        instructions = self.raw_dir / "DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            "# BEA Regional Price Parities\n\n"
            "Download RPP tables from:\n"
            "  https://www.bea.gov/data/prices-inflation/"
            "regional-price-parities-state-and-metro-area\n\n"
            "Or via the interactive tables:\n"
            "  https://apps.bea.gov/iTable/?reqid=70&step=1\n\n"
            "## Recommended tables\n"
            "- SARPP: State all-items RPP\n"
            "- MARPP: Metro area all-items RPP\n\n"
            "Place downloaded files in this directory.\n"
        )
        logger.info("BEA RPP: manual download required. See %s", self.raw_dir)
        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[self.raw_dir],
            source_url="https://www.bea.gov/data/prices-inflation/"
            "regional-price-parities-state-and-metro-area",
            download_date=datetime.now(timezone.utc).isoformat(),
            notes="Manual download required. Instructions written.",
        )
