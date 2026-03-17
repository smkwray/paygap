"""Occupation code harmonization.

Crosswalks Census occupation codes across ACS, CPS, and SIPP
to a common SOC 2018 vintage, and provides coarse groupings.
Also handles O*NET SOC merge keys.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from gender_gap.settings import shared_source_path

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

OCCUPATION_METADATA_COLUMNS = [
    "occupation_title",
    "occupation_title_vintage",
    "soc_major_group",
    "soc_major_label",
]
HARMONIZED_OCCUPATION_COLUMNS = [
    "occupation_code_vintage",
    "survey_year_regime",
    "occupation_code_raw",
    "occupation_title_raw",
    "occupation_title_vintage",
    "occupation_harmonized_code",
    "occupation_harmonized_title",
    "occupation_harmonization_type",
    "soc_major_group",
    "soc_major_label",
]


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


def census_occupation_codebook_path() -> Path:
    """Return the shared canonical path for the official Census occupation workbook."""
    return shared_source_path(
        "census",
        "industry_occupation",
        "2018",
        "2018-occupation-code-list-and-crosswalk.xlsx",
    )


def census_occupation_lookup_path() -> Path:
    """Return the shared canonical path for the normalized occupation lookup CSV."""
    return shared_source_path(
        "census",
        "industry_occupation",
        "2018",
        "census_occupation_code_lookup_2010_2018.csv",
    )


@lru_cache(maxsize=4)
def load_census_harmonized_occupation_lookup(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical harmonized Census occupation lookup.

    The lookup is keyed by raw 4-digit Census occupation codes and maps them to
    a harmonized occupation surface that is stable across 2010/2018 vintages.
    """
    lookup_path = Path(path) if path is not None else census_occupation_lookup_path()
    if lookup_path.exists():
        lookup = pd.read_csv(lookup_path, dtype="string")
        codebook_path = census_occupation_codebook_path()
        if not _lookup_has_year_aware_surface(lookup) and codebook_path.exists():
            lookup = _build_census_harmonized_lookup_from_codebook(codebook_path)
    else:
        codebook_path = census_occupation_codebook_path()
        if not codebook_path.exists():
            return pd.DataFrame(
                columns=[
                    "occupation_code_vintage",
                    "survey_year_regime",
                    "occupation_code_raw",
                    "occupation_title_raw",
                    "occupation_title_vintage",
                    "occupation_harmonized_code",
                    "occupation_harmonized_title",
                    "occupation_harmonization_type",
                    "soc_major_group",
                    "soc_major_label",
                ]
            )
        lookup = _build_census_harmonized_lookup_from_codebook(codebook_path)

    lookup = lookup.copy()
    if "occupation_code_vintage" not in lookup.columns:
        # Backward-compat for older normalized CSVs without vintage dimension.
        lookup["occupation_code_vintage"] = "2018"
    if "survey_year_regime" not in lookup.columns:
        lookup["survey_year_regime"] = "post_2018"
    if "occupation_code_raw" not in lookup.columns and "occupation_code" in lookup.columns:
        # Backward-compat for older normalized CSVs.
        lookup["occupation_code_raw"] = lookup["occupation_code"]
    if "occupation_title_raw" not in lookup.columns and "occupation_title" in lookup.columns:
        lookup["occupation_title_raw"] = lookup["occupation_title"]
    if "occupation_harmonized_code" not in lookup.columns and "occupation_code" in lookup.columns:
        lookup["occupation_harmonized_code"] = lookup["occupation_code"]
    if (
        "occupation_harmonized_title" not in lookup.columns
        and "occupation_title" in lookup.columns
    ):
        lookup["occupation_harmonized_title"] = lookup["occupation_title"]
    if (
        "occupation_harmonization_type" not in lookup.columns
        and "occupation_harmonized_code" in lookup.columns
    ):
        lookup["occupation_harmonization_type"] = "native_2018"
    if "title_vintage" in lookup.columns and "occupation_title_vintage" not in lookup.columns:
        lookup = lookup.rename(columns={"title_vintage": "occupation_title_vintage"})
    lookup["occupation_code_vintage"] = lookup["occupation_code_vintage"].astype("string")
    lookup["occupation_code_raw"] = _normalize_occupation_code(lookup["occupation_code_raw"])
    lookup["occupation_harmonized_code"] = lookup["occupation_harmonized_code"].astype("string")
    for column in HARMONIZED_OCCUPATION_COLUMNS:
        if column not in lookup.columns:
            lookup[column] = pd.NA
    return lookup[HARMONIZED_OCCUPATION_COLUMNS].drop_duplicates(
        ["occupation_code_vintage", "occupation_code_raw"]
    )


@lru_cache(maxsize=4)
def load_census_occupation_lookup(path: str | Path | None = None) -> pd.DataFrame:
    """Backward-compatible lookup keyed by raw occupation code.

    This exposes historical column names while sourcing harmonized titles
    underneath, so existing callers do not break.
    """
    harmonized = load_census_harmonized_occupation_lookup(path).copy()
    legacy = harmonized.rename(
        columns={
            "occupation_code_raw": "occupation_code",
            "occupation_harmonized_title": "occupation_title",
        }
    )[
        [
            "occupation_code",
            "occupation_title",
            "occupation_title_vintage",
            "soc_major_group",
            "soc_major_label",
            "occupation_harmonized_code",
            "occupation_harmonized_title",
            "occupation_harmonization_type",
            "occupation_title_raw",
            "occupation_code_vintage",
            "survey_year_regime",
        ]
    ]
    # This legacy helper cannot preserve vintage collisions on raw code. Prefer
    # the 2018 interpretation when a raw code appears on both surfaces.
    legacy["__sort_vintage"] = legacy["occupation_code_vintage"].map({"2010": 0, "2018": 1})
    legacy = legacy.sort_values(
        ["occupation_code", "__sort_vintage"], ascending=[True, True]
    ).drop_duplicates("occupation_code", keep="last")
    return legacy.drop(columns="__sort_vintage").reset_index(drop=True)


def occupation_code_vintage_from_year(
    survey_year: pd.Series,
    default_vintage: str = "2018",
) -> pd.Series:
    """Infer occupation-code vintage from survey year.

    `2013-2017` map to the 2010 occupation surface. `2018+` map to 2018.
    """
    year = pd.to_numeric(survey_year, errors="coerce")
    vintage = pd.Series(default_vintage, index=survey_year.index, dtype="string")
    vintage.loc[year.notna() & year.le(2017)] = "2010"
    vintage.loc[year.notna() & year.ge(2018)] = "2018"
    return vintage


def harmonize_occupation_codes(
    occupation_code: pd.Series,
    survey_year: pd.Series | None = None,
    path: str | Path | None = None,
    default_vintage: str = "2018",
) -> pd.DataFrame:
    """Harmonize raw occupation codes with optional year-aware vintage handling."""
    lookup = load_census_harmonized_occupation_lookup(path)
    if lookup.empty:
        return pd.DataFrame(columns=HARMONIZED_OCCUPATION_COLUMNS)

    raw = _normalize_occupation_code(occupation_code)
    if survey_year is None:
        vintage = pd.Series(default_vintage, index=raw.index, dtype="string")
    else:
        vintage = occupation_code_vintage_from_year(survey_year, default_vintage=default_vintage)
    keys = pd.DataFrame(
        {
            "occupation_code_vintage": vintage.astype("string"),
            "occupation_code_raw": raw.astype("string"),
        }
    )
    return keys.merge(
        lookup,
        on=["occupation_code_vintage", "occupation_code_raw"],
        how="left",
    )


def attach_harmonized_occupation_metadata(
    df: pd.DataFrame,
    code_col: str = "occupation_code",
    year_col: str = "survey_year",
    path: str | Path | None = None,
    default_vintage: str = "2018",
) -> pd.DataFrame:
    """Append harmonized occupation metadata with year-aware vintage handling."""
    out = df.copy()
    expected_columns = [
        *HARMONIZED_OCCUPATION_COLUMNS,
        "occupation_code",
        "occupation_title",
    ]
    for column in expected_columns:
        if column not in out.columns:
            out[column] = pd.NA
    if code_col not in out.columns:
        return out

    survey_year = out[year_col] if year_col in out.columns else None
    harmonized = harmonize_occupation_codes(
        occupation_code=out[code_col],
        survey_year=survey_year,
        path=path,
        default_vintage=default_vintage,
    )
    if harmonized.empty:
        return out

    merged = out.drop(columns=expected_columns).join(
        harmonized.reset_index(drop=True),
        how="left",
    )
    merged["occupation_code"] = merged["occupation_code_raw"]
    merged["occupation_title"] = merged["occupation_harmonized_title"]
    for column in expected_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged


def attach_occupation_metadata(
    df: pd.DataFrame,
    code_col: str = "occupation_code",
    survey_year_col: str = "survey_year",
    path: str | Path | None = None,
    default_vintage: str = "2018",
) -> pd.DataFrame:
    """Backward-compatible metadata attachment keyed by raw code."""
    return attach_harmonized_occupation_metadata(
        df=df,
        code_col=code_col,
        year_col=survey_year_col,
        path=path,
        default_vintage=default_vintage,
    )


def _build_census_harmonized_lookup_from_codebook(codebook_path: Path) -> pd.DataFrame:
    """Build harmonized occupation mapping from the official Census workbook."""
    list_2018 = _read_2018_occupation_list(codebook_path)
    relations, known_2010 = _read_2010_to_2018_relations(codebook_path)
    return _build_harmonized_lookup(
        list_2018=list_2018,
        relations=relations,
        known_2010=known_2010,
    )


def _read_2018_occupation_list(codebook_path: Path) -> pd.DataFrame:
    codes_2018 = pd.read_excel(
        codebook_path,
        sheet_name="2018 Census Occ Code List",
        header=3,
        dtype="string",
    )
    codes_2018 = codes_2018.iloc[:, 1:4].copy()
    codes_2018.columns = ["occupation_title_2018", "occupation_code", "soc_code_2018"]
    codes_2018["occupation_code"] = _extract_exact_occupation_code(codes_2018["occupation_code"])
    codes_2018 = codes_2018.loc[codes_2018["occupation_code"].notna()].copy()
    codes_2018["occupation_title_2018"] = (
        codes_2018["occupation_title_2018"].astype("string").str.strip()
    )
    codes_2018["soc_code_2018"] = codes_2018["soc_code_2018"].astype("string").str.strip()
    return codes_2018.drop_duplicates("occupation_code")


def _read_2010_to_2018_relations(codebook_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    crosswalk = pd.read_excel(
        codebook_path,
        sheet_name="2010 to 2018 Crosswalk ",
        header=2,
        dtype="string",
    )
    crosswalk = crosswalk.iloc[:, :6].copy()
    crosswalk.columns = [
        "soc_code_2010",
        "occupation_code_2010",
        "occupation_title_2010",
        "soc_code_2018",
        "occupation_code_2018",
        "occupation_title_2018",
    ]
    crosswalk["occupation_code_2010"] = _extract_exact_occupation_code(
        crosswalk["occupation_code_2010"]
    )
    crosswalk["occupation_code_2018"] = _extract_exact_occupation_code(
        crosswalk["occupation_code_2018"]
    )
    # Additional descendants are listed on continuation rows with blank 2010 fields.
    crosswalk[["soc_code_2010", "occupation_code_2010", "occupation_title_2010"]] = crosswalk[
        ["soc_code_2010", "occupation_code_2010", "occupation_title_2010"]
    ].ffill()
    crosswalk["occupation_title_2010"] = (
        crosswalk["occupation_title_2010"].astype("string").str.strip()
    )
    crosswalk["soc_code_2010"] = crosswalk["soc_code_2010"].astype("string").str.strip()
    crosswalk["occupation_title_2018"] = (
        crosswalk["occupation_title_2018"].astype("string").str.strip()
    )
    crosswalk["soc_code_2018"] = crosswalk["soc_code_2018"].astype("string").str.strip()

    known_2010 = crosswalk.loc[crosswalk["occupation_code_2010"].notna(), [
        "occupation_code_2010",
        "occupation_title_2010",
        "soc_code_2010",
    ]].drop_duplicates("occupation_code_2010")
    relations = crosswalk.loc[
        crosswalk["occupation_code_2010"].notna() & crosswalk["occupation_code_2018"].notna(),
        [
            "occupation_code_2010",
            "occupation_title_2010",
            "soc_code_2010",
            "occupation_code_2018",
            "occupation_title_2018",
            "soc_code_2018",
        ],
    ].drop_duplicates(["occupation_code_2010", "occupation_code_2018"])
    return relations, known_2010


def _build_harmonized_lookup(
    list_2018: pd.DataFrame,
    relations: pd.DataFrame,
    known_2010: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    by_2018 = list_2018.set_index("occupation_code")
    desc_groups = relations.groupby("occupation_code_2010", observed=True)["occupation_code_2018"]
    descendants_by_raw = desc_groups.unique().to_dict()
    ancestor_counts = relations.groupby(
        "occupation_code_2018", observed=True
    )["occupation_code_2010"].nunique()
    known_2010_by_code = known_2010.set_index("occupation_code_2010")
    raw_2010_codes = sorted(set(known_2010["occupation_code_2010"].dropna().tolist()))
    for raw_code in raw_2010_codes:
        raw_2010 = (
            known_2010_by_code.loc[raw_code]
            if raw_code in known_2010_by_code.index
            else None
        )
        if raw_2010 is not None:
            raw_title = str(raw_2010["occupation_title_2010"])
            raw_soc = str(raw_2010["soc_code_2010"])
        else:
            raw_title = f"Unknown occupation {raw_code}"
            raw_soc = ""
        descendants = sorted(
            [code for code in descendants_by_raw.get(raw_code, []) if pd.notna(code)]
        )
        if not descendants:
            harmonized_code = f"legacy_2010_{raw_code}"
            harmonized_title = raw_title
            harmonization_type = "legacy_2010_only"
            harmonized_soc = raw_soc
        elif len(descendants) == 1:
            target = descendants[0]
            harmonized_code = target
            if target in by_2018.index:
                harmonized_title = str(by_2018.loc[target]["occupation_title_2018"])
                harmonized_soc = str(by_2018.loc[target]["soc_code_2018"])
            else:
                target_row = relations.loc[relations["occupation_code_2018"] == target].iloc[0]
                harmonized_title = str(target_row["occupation_title_2018"])
                harmonized_soc = str(target_row["soc_code_2018"])
            harmonization_type = (
                "crosswalk_many_to_1"
                if int(ancestor_counts.get(target, 1)) > 1
                else "crosswalk_1_to_1"
            )
        else:
            harmonized_code = f"split_2010_{raw_code}"
            harmonized_title = f"{raw_title} (2010 split bucket)"
            harmonization_type = "crosswalk_1_to_many_split_bucket"
            harmonized_soc = raw_soc
        rows.append(
            _mapping_row(
                vintage="2010",
                regime="pre_2018",
                raw_code=raw_code,
                raw_title=raw_title,
                raw_title_vintage="2010",
                harmonized_code=harmonized_code,
                harmonized_title=harmonized_title,
                harmonization_type=harmonization_type,
                harmonized_soc=harmonized_soc,
            )
        )

    raw_2018_codes = sorted(set(list_2018["occupation_code"].dropna().tolist()))
    for raw_code in raw_2018_codes:
        raw_title = str(by_2018.loc[raw_code]["occupation_title_2018"])
        raw_soc = str(by_2018.loc[raw_code]["soc_code_2018"])
        rows.append(
            _mapping_row(
                vintage="2018",
                regime="post_2018",
                raw_code=raw_code,
                raw_title=raw_title,
                raw_title_vintage="2018",
                harmonized_code=raw_code,
                harmonized_title=raw_title,
                harmonization_type="native_2018",
                harmonized_soc=raw_soc,
            )
        )

    return pd.DataFrame(rows, columns=HARMONIZED_OCCUPATION_COLUMNS).sort_values(
        ["occupation_code_vintage", "occupation_code_raw"]
    )


def _lookup_has_year_aware_surface(lookup: pd.DataFrame) -> bool:
    required = {
        "occupation_code_vintage",
        "occupation_code_raw",
        "occupation_harmonized_code",
        "occupation_harmonized_title",
        "occupation_harmonization_type",
    }
    if not required.issubset(lookup.columns):
        return False
    vintage = lookup["occupation_code_vintage"].astype("string")
    return {"2010", "2018"}.issubset(set(vintage.dropna().tolist()))


def _normalize_occupation_code(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.extract(r"(\d+)")[0]
    return text.where(text.str.fullmatch(r"\d{1,4}", na=False)).str.zfill(4)


def _extract_exact_occupation_code(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.where(text.str.fullmatch(r"\d{4}", na=False))


def _soc_major_group(soc_code: str) -> str:
    if not soc_code:
        return "unknown"
    match = pd.Series([soc_code], dtype="string").str.extract(r"^(\d{2})")[0].iloc[0]
    if pd.isna(match):
        return "unknown"
    return str(match)


def _mapping_row(
    vintage: str,
    regime: str,
    raw_code: str,
    raw_title: str,
    raw_title_vintage: str,
    harmonized_code: str,
    harmonized_title: str,
    harmonization_type: str,
    harmonized_soc: str,
) -> dict[str, str]:
    soc_major_group = _soc_major_group(harmonized_soc)
    return {
        "occupation_code_vintage": vintage,
        "survey_year_regime": regime,
        "occupation_code_raw": raw_code,
        "occupation_title_raw": raw_title,
        "occupation_title_vintage": raw_title_vintage,
        "occupation_harmonized_code": harmonized_code,
        "occupation_harmonized_title": harmonized_title,
        "occupation_harmonization_type": harmonization_type,
        "soc_major_group": soc_major_group,
        "soc_major_label": SOC_MAJOR_GROUPS.get(soc_major_group, "Unknown"),
    }
