"""ATUS (American Time Use Survey) downloader.

Supports:
1. BLS official ATUS files (default)
2. ATUS-X / IPUMS Time Use extracts (placeholder)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from gender_gap.downloaders.base import BaseDownloader, DownloadResult
from gender_gap.settings import DATA_RAW

logger = logging.getLogger(__name__)

# BLS ATUS data files landing page
ATUS_BLS_URL = "https://www.bls.gov/tus/datafiles.htm"
ATUS_IPUMS_URL = "https://www.atusdata.org/"

# Direct download URLs for multi-year BLS ATUS zip bundles
ATUS_FILE_URLS = {
    "respondent": "https://www.bls.gov/tus/datafiles/atusresp-0323.zip",
    "activity": "https://www.bls.gov/tus/datafiles/atusact-0323.zip",
    "activity_summary": "https://www.bls.gov/tus/datafiles/atussum-0323.zip",
    "roster": "https://www.bls.gov/tus/datafiles/atusrost-0323.zip",
    "cps": "https://www.bls.gov/tus/datafiles/atuscps-0323.zip",
}


class ATUSDownloader(BaseDownloader):
    """Download ATUS data from BLS or stage IPUMS ATUS-X extracts."""

    dataset_id = "ATUS"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "atus")

    def download(
        self,
        years: list[int] | None = None,
        use_ipums: bool = False,
        **kwargs,
    ) -> DownloadResult:
        if use_ipums:
            return self._ipums_placeholder(years or [2019, 2021, 2022, 2023])

        return self._download_bls(years)

    def _download_bls(self, years: list[int] | None) -> DownloadResult:
        """Download multi-year BLS ATUS zip files."""
        paths = []
        for name, url in ATUS_FILE_URLS.items():
            dest = self.raw_dir / f"atus_{name}.zip"
            if dest.exists():
                logger.info("ATUS %s already present at %s", name, dest)
                paths.append(dest)
                continue
            logger.info("Downloading ATUS %s from %s", name, url)
            try:
                resp = httpx.get(url, follow_redirects=True, timeout=300)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                paths.append(dest)
                self._log_download(url, dest)
            except httpx.HTTPError as e:
                logger.warning(
                    "Failed to download ATUS %s: %s. "
                    "Manual download may be required from %s",
                    name, e, ATUS_BLS_URL,
                )

        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=paths,
            source_url=ATUS_BLS_URL,
            download_date=datetime.now(timezone.utc).isoformat(),
            notes="BLS multi-year ATUS files. "
            "Filter to desired years during standardization.",
        )

    def _ipums_placeholder(self, years: list[int]) -> DownloadResult:
        """Create placeholder for IPUMS ATUS-X extracts."""
        ipums_dir = self.raw_dir / "ipums"
        ipums_dir.mkdir(parents=True, exist_ok=True)

        instructions = ipums_dir / "DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            "# ATUS-X (IPUMS Time Use) Data Download\n\n"
            f"Create an ATUS-X extract for years {years} at:\n"
            f"  {ATUS_IPUMS_URL}\n\n"
            "## Recommended variables\n"
            "- TUCASEID, TULINENO (identifiers)\n"
            "- SEX, AGE, EDUC, MARST (demographics)\n"
            "- EMPSTAT, FULLPART (employment)\n"
            "- Activity time variables: BLS_PCARE, BLS_CAREHH, BLS_WORK, etc.\n"
            "- TUFINLWGT, TU20FWGT (weights)\n"
            "- Linked CPS earnings variables if available\n"
        )

        logger.info(
            "IPUMS ATUS-X: place your extract files in %s. "
            "Create extracts at %s",
            ipums_dir, ATUS_IPUMS_URL,
        )

        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[ipums_dir],
            source_url=ATUS_IPUMS_URL,
            download_date=datetime.now(timezone.utc).isoformat(),
            notes=f"IPUMS ATUS-X placeholder for years {years}. "
            "Manual extract required.",
        )
