"""Standardize NLSY data for earnings-gap analysis.

NLSY serves as a separate estimation file with cognitive ability (g-factor)
as an additional control. The NLSY cannot be merged onto ACS/CPS individuals,
but the same model specifications can be run on NLSY data to compare results
with and without g-controls.

Data source: processed NLSY files (nlsy79_cfa_resid.csv, nlsy97_cfa_resid.csv)
These contain age-residualized ASVAB subtests from which g_proxy is computed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ASVAB subtests used for g_proxy (unit-weighted composite after z-scoring)
ASVAB_SUBTESTS = ["GS", "AR", "WK", "PC", "NO", "CS", "AS", "MK", "MC", "EI"]

# Path to NLSY processed data (set NLSY_DATA_DIR env var)
NLSY_DATA_DIR = Path(os.environ.get("NLSY_DATA_DIR", "data/external/nlsy"))


def load_nlsy79(
    path: Path | None = None,
    variant: str = "resid",
) -> pd.DataFrame:
    """Load NLSY79 processed data.

    Parameters
    ----------
    path : Path | None
        Path to CSV. If None, uses NLSY_DATA_DIR.
    variant : str
        'resid' for age-residualized (default), '' for raw CFA.
    """
    if path is None:
        suffix = f"_cfa_{variant}" if variant else "_cfa"
        path = NLSY_DATA_DIR / f"nlsy79{suffix}.csv"

    logger.info("Loading NLSY79 from %s", path)
    df = pd.read_csv(path)
    logger.info("NLSY79: %d rows, %d columns", len(df), len(df.columns))
    return df


def load_nlsy97(
    path: Path | None = None,
    variant: str = "resid",
) -> pd.DataFrame:
    """Load NLSY97 processed data."""
    if path is None:
        suffix = f"_cfa_{variant}" if variant else "_cfa"
        path = NLSY_DATA_DIR / f"nlsy97{suffix}.csv"

    logger.info("Loading NLSY97 from %s", path)
    df = pd.read_csv(path)
    logger.info("NLSY97: %d rows, %d columns", len(df), len(df.columns))
    return df


def compute_g_proxy(
    df: pd.DataFrame,
    subtests: list[str] | None = None,
) -> pd.Series:
    """Compute g_proxy as unit-weighted composite of z-scored ASVAB subtests.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the ASVAB subtest columns.
    subtests : list[str] | None
        Subtest column names. If None, uses all 10 standard subtests.

    Returns
    -------
    pd.Series
        Standardized g_proxy (z-scored composite).
    """
    if subtests is None:
        subtests = ASVAB_SUBTESTS

    available = [s for s in subtests if s in df.columns]
    if not available:
        raise ValueError(f"No ASVAB subtests found. Expected: {subtests}")

    missing = [s for s in subtests if s not in df.columns]
    if missing:
        logger.warning("Missing subtests (will be excluded): %s", missing)

    # Z-score each subtest
    z_scores = df[available].apply(lambda x: (x - x.mean()) / x.std())

    # Unit-weighted composite = mean of z-scores
    g_proxy = z_scores.mean(axis=1)

    # Re-standardize the composite
    g_proxy = (g_proxy - g_proxy.mean()) / g_proxy.std()

    return g_proxy


def standardize_nlsy79_for_gap(
    df: pd.DataFrame | None = None,
    cpi_index: dict[int, float] | None = None,
    base_year: int = 2024,
) -> pd.DataFrame:
    """Standardize NLSY79 into a format compatible with paygap models.

    Produces a DataFrame with columns matching the person_year_core schema
    where available, plus g_proxy as the cognitive ability control.

    Uses year-2000 earnings and occupation data (peak career age for this cohort).
    """
    if df is None:
        df = load_nlsy79()

    out = pd.DataFrame()

    # Keys
    out["person_id"] = df["person_id"].astype(str)
    out["data_source"] = "NLSY79"
    out["survey_year"] = 2000
    out["calendar_year"] = 2000

    # Demographics
    out["female"] = (df["sex"] == 2).astype(int)
    out["age"] = df.get("age_2000", df.get("age", pd.NA))
    out["age_sq"] = out["age"] ** 2

    # Race/ethnicity (harmonize to paygap categories)
    out["race_ethnicity"] = _recode_nlsy_race(df["race_ethnicity_3cat"])

    # Education
    out["education_years"] = df.get("education_years", pd.NA)
    ed_yrs = df.get("education_years", pd.Series(dtype=float))
    out["education_level"] = _recode_nlsy_education(ed_yrs)

    # Family
    ms_col = df.get("marital_status_2000", pd.Series(dtype="Int64"))
    out["marital_status"] = _recode_nlsy_marital(ms_col)
    out["number_children"] = df.get("num_children_2000", 0)
    out["children_under_5"] = pd.NA  # Not directly available in NLSY79 processed

    # Job
    out["occupation_code"] = df.get("occupation_code_2000", pd.NA)
    out["class_of_worker"] = pd.NA
    out["self_employed"] = pd.NA

    # Earnings
    earnings = df.get("annual_earnings", pd.Series(dtype=float))
    if cpi_index is not None and 2000 in cpi_index and base_year in cpi_index:
        deflator = cpi_index[base_year] / cpi_index[2000]
        out["annual_earnings_real"] = earnings * deflator
    else:
        out["annual_earnings_real"] = earnings

    # Wealth and household income (useful for mechanism models)
    out["household_income"] = df.get("household_income", pd.NA)
    out["net_worth"] = df.get("net_worth", pd.NA)

    # Parent education (useful control)
    out["parent_education"] = df.get("parent_education", pd.NA)

    # Cognitive ability — the key addition
    out["g_proxy"] = compute_g_proxy(df)

    # Weight
    out["person_weight"] = df.get("sample_weight_96", df.get("sample_weight_79", 1.0))

    return out


def standardize_nlsy97_for_gap(
    df: pd.DataFrame | None = None,
    earnings_year: int = 2019,
    cpi_index: dict[int, float] | None = None,
    base_year: int = 2024,
) -> pd.DataFrame:
    """Standardize NLSY97 into a format compatible with paygap models.

    Uses 2019 or 2021 earnings data (respondents age 35-39 or 37-41).
    """
    if df is None:
        df = load_nlsy97()

    out = pd.DataFrame()

    # Keys
    out["person_id"] = df["person_id"].astype(str)
    out["data_source"] = "NLSY97"
    out["survey_year"] = earnings_year
    out["calendar_year"] = earnings_year

    # Demographics
    out["female"] = (df["sex"] == 2).astype(int)
    age_col = f"age_{earnings_year}"
    out["age"] = df.get(age_col, pd.NA)
    out["age_sq"] = out["age"] ** 2

    out["race_ethnicity"] = _recode_nlsy_race(df["race_ethnicity_3cat"])

    # Education
    out["education_years"] = df.get("education_years", pd.NA)
    ed_yrs97 = df.get("education_years", pd.Series(dtype=float))
    out["education_level"] = _recode_nlsy_education(ed_yrs97)

    # Family
    out["number_children"] = df.get("num_bio_children", 0)
    out["children_under_5"] = pd.NA

    # Job
    occ_col = f"occupation_code_{earnings_year}"
    out["occupation_code"] = df.get(occ_col, pd.NA)

    # Earnings
    earnings_col = f"annual_earnings_{earnings_year}"
    earnings = df.get(earnings_col, pd.Series(dtype=float))
    if cpi_index is not None and earnings_year in cpi_index and base_year in cpi_index:
        deflator = cpi_index[base_year] / cpi_index[earnings_year]
        out["annual_earnings_real"] = earnings * deflator
    else:
        out["annual_earnings_real"] = earnings

    out["household_income"] = df.get(f"household_income_{earnings_year}", pd.NA)
    out["net_worth"] = df.get("net_worth", pd.NA)

    # Parent education
    out["parent_education"] = df.get("parent_education", pd.NA)

    # Cognitive ability
    # NLSY97 uses CAT-ASVAB with POS/NEG scoring; use the computed subtests
    out["g_proxy"] = compute_g_proxy(df)

    # Weight
    out["person_weight"] = df.get("r1_sample_weight", 1.0)

    return out


def _recode_nlsy_race(race_3cat: pd.Series) -> pd.Series:
    """Map NLSY 3-category race to paygap categories."""
    mapping = {
        "NON-BLACK, NON-HISPANIC": "white_non_hispanic",
        "BLACK": "black",
        "HISPANIC": "hispanic",
    }
    return race_3cat.map(mapping).fillna("other")


def _recode_nlsy_education(years: pd.Series) -> pd.Series:
    """Map education years to paygap education levels."""
    result = pd.Series("unknown", index=years.index)
    result[years < 12] = "less_than_hs"
    result[years == 12] = "hs_diploma"
    result[(years > 12) & (years < 16)] = "some_college"
    result[years == 16] = "bachelors"
    result[(years > 16) & (years <= 18)] = "masters"
    result[years > 18] = "doctorate"
    return result


def _recode_nlsy_marital(status: pd.Series) -> pd.Series:
    """Map NLSY79 marital status codes."""
    mapping = {
        0: "never_married",
        1: "married",
        2: "separated",
        3: "divorced",
        6: "widowed",
    }
    return status.map(mapping).fillna("unknown")
