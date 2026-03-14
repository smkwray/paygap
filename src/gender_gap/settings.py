"""Project-wide paths and settings."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

CROSSWALKS_DIR = PROJECT_ROOT / "crosswalks"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CONFIG_CACHE: dict | None = None


def load_config(path: Path | None = None) -> dict:
    """Load the project YAML config, caching after first read."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE
    if path is None:
        path = CONFIGS_DIR / "datasets.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if path == CONFIGS_DIR / "datasets.yaml":
        _CONFIG_CACHE = cfg
    return cfg


# Default base year for inflation adjustment
BASE_CURRENCY_YEAR = 2024

# ---------------------------------------------------------------------------
# API keys (loaded from environment / .env)
# ---------------------------------------------------------------------------


def get_api_key(service: str) -> str | None:
    """Get an API key from environment variables.

    Supported services: bea, fred, bls, census, ipums, noaa, usda
    """
    env_map = {
        "bea": "BEA_API_KEY",
        "fred": "FRED_API_KEY",
        "bls": "BLS_API_KEY",
        "census": "CENSUS_API_KEY",
        "ipums": "IPUMS_API_KEY",
        "noaa": "NOAA_API_KEY",
        "usda": "USDA_API_KEY",
    }
    env_var = env_map.get(service.lower())
    if env_var is None:
        return None
    return os.environ.get(env_var)


# Load .env file if python-dotenv is available, otherwise keys come from shell env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
