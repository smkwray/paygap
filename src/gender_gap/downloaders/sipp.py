"""SIPP (Survey of Income and Program Participation) downloader.

SIPP public-use files are available from Census. Variable names vary by panel
year and topical module, so the standardizer must consult year-specific
codebooks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from gender_gap.downloaders.base import BaseDownloader, DownloadResult
from gender_gap.settings import DATA_RAW

logger = logging.getLogger(__name__)

# Census SIPP data download landing page
SIPP_DATASETS_URL = "https://www.census.gov/programs-surveys/sipp/data/datasets.html"


class SIPPDownloader(BaseDownloader):
    """Download SIPP public-use microdata from Census."""

    dataset_id = "SIPP"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "sipp")

    def download(
        self,
        years: list[int] | None = None,
        **kwargs,
    ) -> DownloadResult:
        """Stage SIPP data.

        SIPP public-use files require manual download from Census because
        the URLs change with each panel release and file format varies.
        This method creates the directory structure and instructions.
        """
        if years is None:
            years = [2022, 2023]

        instructions = self.raw_dir / "DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            "# SIPP Data Download\n\n"
            f"Download SIPP public-use data for years {years} from:\n"
            f"  {SIPP_DATASETS_URL}\n\n"
            "Place the downloaded files in this directory.\n\n"
            "## Recommended files\n"
            "- Person-level interview files (SAS or Stata format)\n"
            "- Use `pyreadstat` to read SAS (.sas7bdat) or Stata (.dta) files\n\n"
            "## Variable name notes\n"
            "SIPP variable names depend on panel year and topical module.\n"
            "The standardizer will use year-specific codebook mappings.\n"
            "Key variables to ensure are present:\n"
            "  - Demographics: ESEX, TAGE, EEDUC\n"
            "  - Employment: RMESR, EJBHRS1, TJBHRS1\n"
            "  - Earnings: TJB1_MSUM, TJB1_ANNSAL1\n"
            "  - Commute: TJB1_PVTIME, TJB1_PVMILE, EJB1_PVTRPRM\n"
            "  - Family: THHLDSTATUS, TCHILD, TAGE_FB\n"
            "  - Union: EJB1_UNION\n"
        )

        logger.info(
            "SIPP: place downloaded files in %s. "
            "See DOWNLOAD_INSTRUCTIONS.md for details.",
            self.raw_dir,
        )

        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[self.raw_dir],
            source_url=SIPP_DATASETS_URL,
            download_date=datetime.now(timezone.utc).isoformat(),
            notes=f"Manual download required for years {years}. "
            "Instructions written to DOWNLOAD_INSTRUCTIONS.md.",
        )
