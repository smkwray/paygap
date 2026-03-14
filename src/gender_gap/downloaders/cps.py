"""CPS Basic Monthly / ORG downloader.

Supports:
1. IPUMS CPS extracts (recommended)
2. Official Census/BLS raw monthly files
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from gender_gap.downloaders.base import BaseDownloader, DownloadResult
from gender_gap.settings import DATA_RAW

logger = logging.getLogger(__name__)


class CPSDownloader(BaseDownloader):
    """Download CPS Basic Monthly / ORG data."""

    dataset_id = "CPS_ORG_BASIC_MONTHLY"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "cps")

    def download(
        self,
        years: list[int] | None = None,
        use_ipums: bool = True,
        **kwargs,
    ) -> DownloadResult:
        if years is None:
            years = [2022, 2023]

        if use_ipums:
            return self._ipums_placeholder(years)

        # Official raw monthly files from Census
        # Format varies by year; implementation would need per-year URL logic
        logger.warning(
            "Official CPS raw download not yet implemented. "
            "Use IPUMS CPS (use_ipums=True) or place files in %s",
            self.raw_dir,
        )
        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[self.raw_dir],
            source_url="https://www.census.gov/data/datasets/time-series/demo/cps/cps-basic.html",
            download_date=datetime.now(timezone.utc).isoformat(),
            notes=f"Placeholder for years {years}. Manual download required.",
        )

    def _ipums_placeholder(self, years: list[int]) -> DownloadResult:
        ipums_dir = self.raw_dir / "ipums"
        ipums_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "IPUMS CPS: place your extract files in %s. "
            "Create extracts at https://cps.ipums.org/cps/",
            ipums_dir,
        )
        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[ipums_dir],
            source_url="https://cps.ipums.org/cps/",
            download_date=datetime.now(timezone.utc).isoformat(),
            notes=f"IPUMS CPS placeholder for years {years}. Manual extract required.",
        )
