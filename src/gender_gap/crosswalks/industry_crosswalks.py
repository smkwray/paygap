"""Industry code harmonization.

Crosswalks Census industry codes to NAICS 2-digit sectors
and provides broad industry groupings for analysis.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# NAICS 2-digit sector labels
NAICS_SECTORS = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation and Warehousing",
    "49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
    "99": "Military",
}

# Broad industry categories
BROAD_IND_MAP = {
    "Agriculture, Forestry, Fishing and Hunting": "natural_resources",
    "Mining, Quarrying, and Oil and Gas Extraction": "natural_resources",
    "Utilities": "natural_resources",
    "Construction": "construction",
    "Manufacturing": "manufacturing",
    "Wholesale Trade": "trade",
    "Retail Trade": "trade",
    "Transportation and Warehousing": "transport_utilities",
    "Information": "information_finance",
    "Finance and Insurance": "information_finance",
    "Real Estate and Rental and Leasing": "information_finance",
    "Professional, Scientific, and Technical Services": "professional_services",
    "Management of Companies and Enterprises": "professional_services",
    "Administrative and Support and Waste Management": "professional_services",
    "Educational Services": "education_health",
    "Health Care and Social Assistance": "education_health",
    "Arts, Entertainment, and Recreation": "leisure_hospitality",
    "Accommodation and Food Services": "leisure_hospitality",
    "Other Services (except Public Administration)": "other_services",
    "Public Administration": "government",
    "Military": "military",
}


def census_ind_to_naics2(ind_code: pd.Series) -> pd.Series:
    """Map Census industry codes to approximate NAICS 2-digit sector.

    ACS INDP and CPS IND codes map approximately to NAICS.

    Parameters
    ----------
    ind_code : pd.Series
        Census industry code (4-digit numeric).

    Returns
    -------
    pd.Series
        NAICS 2-digit sector code.
    """
    code_num = pd.to_numeric(ind_code, errors="coerce")
    result = pd.Series("unknown", index=ind_code.index)

    # Census industry code ranges → NAICS 2-digit
    ranges = [
        (170, 290, "11"), (370, 490, "21"), (570, 690, "22"),
        (770, 770, "23"), (1070, 3990, "31"),  # Manufacturing
        (4070, 4590, "42"), (4670, 5790, "44"),  # Retail
        (6070, 6390, "48"), (6470, 6780, "51"),
        (6870, 6990, "52"), (7070, 7190, "53"),
        (7270, 7490, "54"), (7570, 7580, "55"),
        (7590, 7790, "56"), (7860, 7890, "61"),
        (7970, 8470, "62"), (8560, 8590, "71"),
        (8660, 8690, "72"), (8770, 9290, "81"),
        (9370, 9590, "92"), (9670, 9890, "99"),
    ]

    for lo, hi, naics2 in ranges:
        result[(code_num >= lo) & (code_num <= hi)] = naics2

    return result


def naics2_to_label(naics2: pd.Series) -> pd.Series:
    """Map NAICS 2-digit codes to human-readable sector labels."""
    return naics2.map(NAICS_SECTORS).fillna("Unknown")


def naics2_to_broad(naics2: pd.Series) -> pd.Series:
    """Map NAICS 2-digit to broad industry categories."""
    labels = naics2.map(NAICS_SECTORS)
    return labels.map(BROAD_IND_MAP).fillna("other")
