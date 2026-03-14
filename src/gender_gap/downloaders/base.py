"""Base downloader interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a download operation."""

    dataset_id: str
    paths: list[Path]
    source_url: str
    download_date: str
    notes: str = ""


class BaseDownloader(ABC):
    """Abstract base for dataset downloaders."""

    dataset_id: str

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self, years: list[int] | None = None, **kwargs) -> DownloadResult:
        """Download raw data files. Return a DownloadResult."""
        ...

    def _log_download(self, url: str, dest: Path) -> None:
        logger.info("Downloaded %s -> %s", url, dest)
