"""Project-wide paths and settings."""

from __future__ import annotations

import os
from pathlib import Path
from string import Template

from gender_gap.utils.yaml_compat import load_yaml

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_DATA_ROOT = PROJECT_ROOT.parent / "data"
SHARED_DATA_ROOT = Path(
    os.environ.get("PROJ_SHARED_DATA_ROOT", str(DEFAULT_SHARED_DATA_ROOT))
).expanduser()

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

CROSSWALKS_DIR = PROJECT_ROOT / "crosswalks"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

SHARED_SOURCES_DIR = SHARED_DATA_ROOT / "sources"
SHARED_CATALOG_DIR = SHARED_DATA_ROOT / "catalog"
SHARED_DATASETS_CATALOG = SHARED_CATALOG_DIR / "datasets.csv"
SHARED_ALIASES_CATALOG = SHARED_CATALOG_DIR / "aliases.csv"

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
        raw = f.read()
    expanded = Template(raw).safe_substitute(os.environ)
    cfg = load_yaml(expanded) or {}
    if path == CONFIGS_DIR / "datasets.yaml":
        _CONFIG_CACHE = cfg
    return cfg


def load_repro_config(path: Path | None = None) -> dict:
    """Load the reproductive-burden extension config."""
    if path is None:
        path = CONFIGS_DIR / "repro_extension.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = f.read()
    expanded = Template(raw).safe_substitute(os.environ)
    return load_yaml(expanded) or {}


def load_variance_config(path: Path | None = None) -> dict:
    """Load the variance addon config."""
    if path is None:
        path = CONFIGS_DIR / "variance_addon.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = f.read()
    expanded = Template(raw).safe_substitute(os.environ)
    return load_yaml(expanded) or {}


def shared_source_path(*parts: str) -> Path:
    """Build a canonical shared-source path under the project data root."""
    return SHARED_SOURCES_DIR.joinpath(*parts)


def shared_catalog_path(name: str) -> Path:
    """Return a file path inside the shared catalog directory."""
    return SHARED_CATALOG_DIR / name


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
