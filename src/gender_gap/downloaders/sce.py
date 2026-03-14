"""New York Fed SCE Labor Market Survey downloader.

This is a supplemental expectations dataset rather than a main wage-regression
file. Public access is through the New York Fed SCE data portal/databank and
released microdata lag the survey by roughly 18 months.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from gender_gap.downloaders.base import BaseDownloader, DownloadResult
from gender_gap.settings import DATA_RAW

logger = logging.getLogger(__name__)

SCE_MAIN_URL = "https://www.newyorkfed.org/microeconomics/sce"
SCE_DATABANK_URL = "https://www.newyorkfed.org/microeconomics/databank.html"
SCE_FAQ_URL = "https://www.newyorkfed.org/microeconomics/sce/sce-faq"
SCE_LABOR_URL = "https://www.newyorkfed.org/microeconomics/sce/labor"
SCE_QUESTIONNAIRE_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/microeconomics/"
    "interactive/downloads/sce-labor-questionnaire.pdf"
)
SCE_CODEBOOK_URL = (
    "https://www.newyorkfed.org/medialibrary/Interactives/sce/sce/downloads/data/"
    "SCE-Labor-Market-Survey-Data-Codebook.pdf?sc_lang=en"
)
SCE_RESERVATION_WAGE_POST_URL = (
    "https://libertystreeteconomics.newyorkfed.org/2024/08/"
    "an-update-on-the-reservation-wages-in-the-sce-labor-market-survey/"
)


class SCELaborMarketDownloader(BaseDownloader):
    """Stage the NY Fed SCE labor-market expectations data."""

    dataset_id = "SCE_LABOR_MARKET"

    def __init__(self, raw_dir: Path | None = None):
        super().__init__(raw_dir or DATA_RAW / "sce_labor_market")

    def download(
        self,
        years: list[int] | None = None,
        **kwargs,
    ) -> DownloadResult:
        """Write acquisition notes for the SCE labor-market files.

        The public SCE release is not a stable direct-download endpoint in this
        repo, so the downloader stages instructions and the target directory.
        """
        instructions = self.raw_dir / "DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            "# NY Fed SCE Labor Market Survey\n\n"
            "Use this as a supplemental expectations/mechanism dataset.\n"
            "Do not merge it person-by-person onto ACS/CPS/SIPP.\n\n"
            "## Official sources\n"
            f"- SCE overview: {SCE_MAIN_URL}\n"
            f"- SCE Labor Market page: {SCE_LABOR_URL}\n"
            f"- SCE databank and microdata access: {SCE_DATABANK_URL}\n\n"
            f"- SCE FAQ: {SCE_FAQ_URL}\n"
            f"- SCE questionnaire: {SCE_QUESTIONNAIRE_URL}\n"
            f"- SCE codebook: {SCE_CODEBOOK_URL}\n"
            f"- New York Fed reservation wage note: {SCE_RESERVATION_WAGE_POST_URL}\n\n"
            "## Why it matters here\n"
            "- Reservation wage (lowest wage accepted for a new job)\n"
            "- Wage expectations and expected wage growth\n"
            "- Realized offer wages and offer acceptance/rejection outcomes\n\n"
            "## Use in this repo\n"
            "- Supporting evidence on bargaining-related channels\n"
            "- Calibration/comparison against realized wage-gap estimates\n"
            "- Not a literal control variable in the main ACS/CPS wage regressions\n\n"
            "## Survey design notes\n"
            "- Labor Market Survey first fielded in March 2014; first public release August 2017\n"
            "- Fielded every four months, with data collection in March, July, and November\n"
            "- About 1,000 respondents per wave; weighted to represent U.S. household heads\n"
            "- Public microdata and chart downloads live on the New York Fed Data Bank\n\n"
            "## Release notes\n"
            "- Public microdata are separate from ACS/CPS/ATUS and should be analyzed as a supplemental module\n"
            "- If you acquire microdata, keep the raw files under data/raw/sce_labor_market/ and build report artifacts rather than attempting person-level merges\n"
        )

        logger.info(
            "SCE Labor Market Survey: review %s and place any downloaded files in %s",
            instructions,
            self.raw_dir,
        )

        return DownloadResult(
            dataset_id=self.dataset_id,
            paths=[self.raw_dir, instructions],
            source_url=SCE_DATABANK_URL,
            download_date=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Supplemental expectations dataset for reservation wages and job offers. "
                "Manual download required from the New York Fed SCE portal."
            ),
        )
