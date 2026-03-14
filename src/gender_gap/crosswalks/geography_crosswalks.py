"""Geography crosswalks: PUMA, CBSA, state, county mappings.

Provides geographic fallback hierarchy for contextual merges
and standardized state/metro identifiers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gender_gap.settings import CROSSWALKS_DIR

logger = logging.getLogger(__name__)

PUMA_CBSA_CROSSWALK_PATH = CROSSWALKS_DIR / "geography" / "puma_to_cbsa_crosswalk.csv"
_PUMA_CBSA_CACHE: dict[Path, pd.DataFrame | None] = {}

# State FIPS to abbreviation
STATE_FIPS_TO_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT",
    10: "DE", 11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL",
    18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD",
    25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
    32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND",
    39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD",
    47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY", 72: "PR",
}

# Census region mapping
STATE_TO_REGION = {
    "CT": "NE", "ME": "NE", "MA": "NE", "NH": "NE", "RI": "NE", "VT": "NE",
    "NJ": "NE", "NY": "NE", "PA": "NE",
    "IL": "MW", "IN": "MW", "MI": "MW", "OH": "MW", "WI": "MW",
    "IA": "MW", "KS": "MW", "MN": "MW", "MO": "MW", "NE": "MW",
    "ND": "MW", "SD": "MW",
    "DE": "SO", "FL": "SO", "GA": "SO", "MD": "SO", "NC": "SO",
    "SC": "SO", "VA": "SO", "DC": "SO", "WV": "SO",
    "AL": "SO", "KY": "SO", "MS": "SO", "TN": "SO",
    "AR": "SO", "LA": "SO", "OK": "SO", "TX": "SO",
    "AZ": "WE", "CO": "WE", "ID": "WE", "MT": "WE", "NV": "WE",
    "NM": "WE", "UT": "WE", "WY": "WE",
    "AK": "WE", "CA": "WE", "HI": "WE", "OR": "WE", "WA": "WE",
    "PR": "PR",
}

REGION_LABELS = {
    "NE": "Northeast",
    "MW": "Midwest",
    "SO": "South",
    "WE": "West",
    "PR": "Puerto Rico",
}


def state_fips_to_abbr(fips: pd.Series) -> pd.Series:
    """Convert state FIPS codes to 2-letter abbreviations."""
    return pd.to_numeric(fips, errors="coerce").map(STATE_FIPS_TO_ABBR).fillna("UNK")


def state_fips_to_region(fips: pd.Series) -> pd.Series:
    """Map state FIPS to Census region code (NE, MW, SO, WE)."""
    abbr = state_fips_to_abbr(fips)
    return abbr.map(STATE_TO_REGION).fillna("UNK")


def state_fips_to_region_label(fips: pd.Series) -> pd.Series:
    """Map state FIPS to Census region name."""
    region = state_fips_to_region(fips)
    return region.map(REGION_LABELS).fillna("Unknown")


def build_geo_merge_key(
    df: pd.DataFrame,
    fallback_order: list[str] | None = None,
) -> pd.Series:
    """Build a geographic merge key using fallback hierarchy.

    Uses the most specific geography available for each row:
    county > cbsa > puma > state > national.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain some subset of geography columns.
    fallback_order : list[str] | None
        Column names in priority order. Default: county_fips, cbsa_code,
        residence_puma (combined with state), state_fips.

    Returns
    -------
    pd.Series
        Geographic merge key string, e.g., "county:06037" or "state:06".
    """
    if fallback_order is None:
        fallback_order = ["county_fips", "cbsa_code", "residence_puma", "state_fips"]

    result = pd.Series("national:US", index=df.index)

    # Apply in reverse order so most specific wins
    for col in reversed(fallback_order):
        if col not in df.columns:
            continue
        valid = df[col].notna()
        if col == "residence_puma" and "state_fips" in df.columns:
            # PUMA is only unique within state
            key = "puma:" + df["state_fips"].astype(str) + "_" + df[col].astype(str)
        else:
            prefix = col.replace("_fips", "").replace("_code", "")
            key = prefix + ":" + df[col].astype(str)
        result[valid] = key[valid]

    return result


def load_puma_cbsa_crosswalk(
    path: Path | None = None,
    force_reload: bool = False,
) -> pd.DataFrame | None:
    """Load the repo-local PUMA-to-CBSA crosswalk if present."""
    if path is None:
        path = PUMA_CBSA_CROSSWALK_PATH
    path = Path(path)

    if not force_reload and path in _PUMA_CBSA_CACHE:
        return _PUMA_CBSA_CACHE[path]
    if not path.exists():
        _PUMA_CBSA_CACHE[path] = None
        return None

    df = pd.read_csv(path, dtype={"state_fips": "string", "puma": "string", "cbsa_code": "string"})
    df["state_fips"] = df["state_fips"].str.zfill(2)
    df["puma"] = df["puma"].str.zfill(5)
    _PUMA_CBSA_CACHE[path] = df
    return df


def append_puma_cbsa_crosswalk(
    df: pd.DataFrame,
    state_col: str = "state_fips",
    puma_col: str = "residence_puma",
    path: Path | None = None,
) -> pd.DataFrame:
    """Append dominant CBSA and metro status using state+PUMA keys.

    When the crosswalk is unavailable, append placeholder columns filled with
    `unknown` / `NA` instead of failing.
    """
    result = df.copy()
    crosswalk = load_puma_cbsa_crosswalk(path=path)
    if crosswalk is None:
        result["cbsa_code"] = pd.Series(pd.NA, index=result.index, dtype="string")
        result["cbsa_title"] = pd.Series(pd.NA, index=result.index, dtype="string")
        result["metro_status"] = "unknown"
        return result

    merge_keys = pd.DataFrame(index=result.index)
    merge_keys["state_fips"] = _normalize_state_fips(result.get(state_col), index=result.index)
    merge_keys["puma"] = _normalize_puma(result.get(puma_col), index=result.index)
    merge_keys["__row_id"] = result.index

    merged = merge_keys.merge(
        crosswalk[["state_fips", "puma", "cbsa_code", "cbsa_title", "metro_status"]],
        how="left",
        on=["state_fips", "puma"],
    ).set_index("__row_id")

    result["cbsa_code"] = merged["cbsa_code"].astype("string")
    result["cbsa_title"] = merged["cbsa_title"].astype("string")
    result["metro_status"] = merged["metro_status"].fillna("unknown")
    return result


def metro_indicator(
    state_fips: pd.Series,
    puma: pd.Series | None = None,
    path: Path | None = None,
) -> pd.Series:
    """Return metro status using the dominant PUMA-to-CBSA crosswalk when available."""
    if puma is None:
        return pd.Series("unknown", index=state_fips.index)
    df = pd.DataFrame({"state_fips": state_fips, "residence_puma": puma})
    return append_puma_cbsa_crosswalk(df, path=path)["metro_status"]


def _normalize_state_fips(values, index) -> pd.Series:
    if values is None:
        return pd.Series(pd.NA, index=index, dtype="string")
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("Int64").astype("string").str.zfill(2)


def _normalize_puma(values, index) -> pd.Series:
    if values is None:
        return pd.Series(pd.NA, index=index, dtype="string")
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("Int64").astype("string").str.zfill(5)
