"""Occupation code harmonization.

Crosswalks Census occupation codes across ACS, CPS, and SIPP
to a common SOC 2018 vintage, and provides coarse groupings.
Also handles O*NET SOC merge keys.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Census 2018 SOC major groups (2-digit)
SOC_MAJOR_GROUPS = {
    "11": "Management",
    "13": "Business and Financial Operations",
    "15": "Computer and Mathematical",
    "17": "Architecture and Engineering",
    "19": "Life, Physical, and Social Science",
    "21": "Community and Social Service",
    "23": "Legal",
    "25": "Educational Instruction and Library",
    "27": "Arts, Design, Entertainment, Sports, Media",
    "29": "Healthcare Practitioners and Technical",
    "31": "Healthcare Support",
    "33": "Protective Service",
    "35": "Food Preparation and Serving",
    "37": "Building and Grounds Cleaning and Maintenance",
    "39": "Personal Care and Service",
    "41": "Sales and Related",
    "43": "Office and Administrative Support",
    "45": "Farming, Fishing, and Forestry",
    "47": "Construction and Extraction",
    "49": "Installation, Maintenance, and Repair",
    "51": "Production",
    "53": "Transportation and Material Moving",
    "55": "Military Specific",
}

# Broad occupation categories for analysis
BROAD_OCC_MAP = {
    "Management": "management_professional",
    "Business and Financial Operations": "management_professional",
    "Computer and Mathematical": "management_professional",
    "Architecture and Engineering": "management_professional",
    "Life, Physical, and Social Science": "management_professional",
    "Community and Social Service": "management_professional",
    "Legal": "management_professional",
    "Educational Instruction and Library": "management_professional",
    "Arts, Design, Entertainment, Sports, Media": "management_professional",
    "Healthcare Practitioners and Technical": "healthcare",
    "Healthcare Support": "healthcare",
    "Protective Service": "service",
    "Food Preparation and Serving": "service",
    "Building and Grounds Cleaning and Maintenance": "service",
    "Personal Care and Service": "service",
    "Sales and Related": "sales_office",
    "Office and Administrative Support": "sales_office",
    "Farming, Fishing, and Forestry": "natural_resources_construction",
    "Construction and Extraction": "natural_resources_construction",
    "Installation, Maintenance, and Repair": "production_transport",
    "Production": "production_transport",
    "Transportation and Material Moving": "production_transport",
    "Military Specific": "military",
}


def census_occ_to_soc_major(occ_code: pd.Series) -> pd.Series:
    """Map Census occupation codes to SOC 2-digit major group.

    ACS OCCP and CPS OCC codes map approximately to SOC codes.
    This uses the first 2 digits of the 4-digit Census code
    to approximate the SOC major group.

    Parameters
    ----------
    occ_code : pd.Series
        Census occupation code (4-digit numeric or string).

    Returns
    -------
    pd.Series
        SOC major group code (2-digit string).
    """
    occ_code.astype(str).str.zfill(4)
    # Census occupation codes: first 2 digits approximate SOC major group
    # 0010-0440 → 11 (Management)
    # 0500-0960 → 13 (Business/Financial)
    # etc.
    # Use a simplified mapping based on code ranges
    code_num = pd.to_numeric(occ_code, errors="coerce")
    result = pd.Series("unknown", index=occ_code.index)

    ranges = [
        (10, 440, "11"), (500, 960, "13"), (1005, 1240, "15"),
        (1305, 1560, "17"), (1600, 1980, "19"), (2001, 2060, "21"),
        (2100, 2180, "23"), (2200, 2555, "25"), (2600, 2970, "27"),
        (3000, 3550, "29"), (3600, 3655, "31"), (3700, 3960, "33"),
        (4000, 4160, "35"), (4200, 4255, "37"), (4330, 4655, "39"),
        (4700, 4965, "41"), (5000, 5940, "43"), (6005, 6130, "45"),
        (6200, 6950, "47"), (7000, 7640, "49"), (7700, 8990, "51"),
        (9000, 9760, "53"), (9800, 9840, "55"),
    ]

    for lo, hi, soc2 in ranges:
        result[(code_num >= lo) & (code_num <= hi)] = soc2

    return result


def soc_major_to_label(soc2: pd.Series) -> pd.Series:
    """Map SOC 2-digit codes to human-readable labels."""
    return soc2.map(SOC_MAJOR_GROUPS).fillna("Unknown")


def soc_major_to_broad(soc2: pd.Series) -> pd.Series:
    """Map SOC 2-digit codes to broad occupation categories."""
    labels = soc2.map(SOC_MAJOR_GROUPS)
    return labels.map(BROAD_OCC_MAP).fillna("other")


def onet_soc_to_census_soc(onet_soc: pd.Series) -> pd.Series:
    """Convert O*NET-SOC codes (XX-XXXX.XX) to 2-digit SOC major group.

    O*NET codes are formatted as 'XX-XXXX.XX'. The first two digits
    are the SOC major group.
    """
    return onet_soc.astype(str).str[:2]
