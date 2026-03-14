"""Dataset registry: reads DATASET_REGISTRY.csv and provides structured access."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from gender_gap.settings import CONFIGS_DIR


@dataclass
class DatasetEntry:
    """One row from DATASET_REGISTRY.csv."""

    dataset_id: str
    category: str
    priority: str
    recommended_use: str
    why_it_matters: str
    official_url: str
    official_access_url: str
    api_or_programmatic_url: str
    faster_alt_url: str
    formats: str
    geography: str
    unit_of_analysis: str
    refresh: str
    weights: str
    join_keys: str
    must_have_fields: str
    cautions: str

    @property
    def is_core(self) -> bool:
        return self.priority in ("core", "core_support")


_REGISTRY_CACHE: list[DatasetEntry] | None = None


def load_registry(path: Path | None = None) -> list[DatasetEntry]:
    """Load the dataset registry from CSV."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None and path is None:
        return _REGISTRY_CACHE

    if path is None:
        path = CONFIGS_DIR / "DATASET_REGISTRY.csv"

    entries: list[DatasetEntry] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                DatasetEntry(
                    dataset_id=row.get("dataset_id", ""),
                    category=row.get("category", ""),
                    priority=row.get("priority", ""),
                    recommended_use=row.get("recommended_use", ""),
                    why_it_matters=row.get("why_it_matters", ""),
                    official_url=row.get("official_url", ""),
                    official_access_url=row.get("official_access_url", ""),
                    api_or_programmatic_url=row.get("api_or_programmatic_url", ""),
                    faster_alt_url=row.get("faster_alt_url", ""),
                    formats=row.get("formats", ""),
                    geography=row.get("geography", ""),
                    unit_of_analysis=row.get("unit_of_analysis", ""),
                    refresh=row.get("refresh", ""),
                    weights=row.get("weights", ""),
                    join_keys=row.get("join_keys", ""),
                    must_have_fields=row.get("must_have_fields", ""),
                    cautions=row.get("cautions", ""),
                )
            )

    if path == CONFIGS_DIR / "DATASET_REGISTRY.csv":
        _REGISTRY_CACHE = entries

    return entries


def get_dataset(dataset_id: str) -> DatasetEntry | None:
    """Look up a single dataset by ID."""
    for entry in load_registry():
        if entry.dataset_id == dataset_id:
            return entry
    return None


def core_datasets() -> list[DatasetEntry]:
    """Return only core and core_support datasets."""
    return [e for e in load_registry() if e.is_core]


def print_registry() -> None:
    """Print a summary table of the registry."""
    entries = load_registry()
    print(f"{'ID':<30} {'Priority':<15} {'Category':<12} {'Use'}")
    print("-" * 100)
    for e in entries:
        use = e.recommended_use
        use_short = use[:50] + "..." if len(use) > 50 else use
        print(f"{e.dataset_id:<30} {e.priority:<15} {e.category:<12} {use_short}")
