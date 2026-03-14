#!/usr/bin/env python3
"""Run full model stack on ALL downloaded datasets.

Expects data/raw/ to be populated by download_all.py.

Processes:
  1. ACS PUMS 2015-2023: standardize → features → full model stack each year
  2. ACS pooled panel (2015-2023): year fixed effects, trend analysis
  3. CPS ASEC 2015-2023: standardize → features → OLS/descriptive each year
  4. ATUS: standardize → time-use mechanism analysis
  5. Context data: merge → contextual controls for ACS/CPS
  6. Cross-dataset comparison and trend reporting
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_all")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CPI-U annual averages for deflation to 2022 dollars
CPI_U = {
    2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957,
    2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.120,
    2018: 251.107, 2019: 255.657, 2020: 258.811, 2021: 270.970,
    2022: 292.655, 2023: 304.702, 2024: 312.332,
}
BASE_YEAR = 2022
MAX_REASONABLE_CPS_EARNINGS = 2_000_000


def deflate(value, from_year):
    """Deflate nominal value to BASE_YEAR dollars."""
    if from_year in CPI_U and BASE_YEAR in CPI_U:
        return value * CPI_U[BASE_YEAR] / CPI_U[from_year]
    return value


def _clean_cps_earnings(series: pd.Series) -> pd.Series:
    """Remove malformed CPS earnings values while preserving topcoded highs."""
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.mask(cleaned < 0, 0)
    cleaned = cleaned.mask(cleaned > MAX_REASONABLE_CPS_EARNINGS, np.nan)
    return cleaned


# ═══════════════════════════════════════════════════════════════════
# ACS STANDARDIZATION & ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def _acs_raw_path(year: int, variant: str = "api") -> Path:
    if variant == "api":
        return DATA_RAW / "acs" / f"acs_pums_{year}_api.parquet"
    if variant == "api_repweights":
        return DATA_RAW / "acs" / f"acs_pums_{year}_api_repweights.parquet"
    raise ValueError(f"Unsupported ACS raw variant: {variant}")


def standardize_acs_year(year: int, raw_variant: str = "api") -> pd.DataFrame | None:
    """Standardize a single ACS year and build analysis features."""
    processed_suffix = "_repweights" if raw_variant == "api_repweights" else ""
    processed_path = DATA_PROCESSED / f"acs_{year}_analysis_ready{processed_suffix}.parquet"
    if processed_path.exists():
        logger.info("ACS %d: analysis-ready cached", year)
        return pd.read_parquet(processed_path)

    raw_path = _acs_raw_path(year, variant=raw_variant)
    if not raw_path.exists():
        logger.warning("ACS %d: raw data not found at %s", year, raw_path)
        return None

    from gender_gap.standardize.acs_standardize import standardize_acs
    from gender_gap.features.earnings import winsorize_wages, log_wage
    from gender_gap.features.sample_filters import (
        filter_prime_age_wage_salary, drop_outlier_wages,
    )
    from gender_gap.crosswalks.occupation_crosswalks import (
        census_occ_to_soc_major, soc_major_to_broad,
    )
    from gender_gap.crosswalks.industry_crosswalks import (
        census_ind_to_naics2, naics2_to_broad,
    )
    from gender_gap.crosswalks.geography_crosswalks import (
        append_puma_cbsa_crosswalk, state_fips_to_abbr, state_fips_to_region,
    )

    logger.info("ACS %d: standardizing...", year)
    df_raw = pd.read_parquet(raw_path)

    # Ensure numeric
    for col in ["SEX", "AGEP", "WAGP", "PERNP", "WKHP", "WKWN", "WKW", "COW",
                 "JWTRNS", "JWMNP", "ST", "PWGTP", "MAR", "ADJINC", "HISP",
                 "SCHL", "RAC1P", "SPORDER", "OCCP", "INDP"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Fix ADJINC
    if "ADJINC" in df_raw.columns and df_raw["ADJINC"].max() < 100:
        df_raw["ADJINC"] = df_raw["ADJINC"] * 1_000_000

    keep_replicate_weights = raw_variant == "api_repweights"
    df = standardize_acs(
        df_raw,
        survey_year=year,
        keep_replicate_weights=keep_replicate_weights,
    )
    df["hourly_wage_real"] = pd.to_numeric(df["hourly_wage_real"], errors="coerce")

    # Sample filter
    n_before = len(df)
    df = filter_prime_age_wage_salary(df)
    logger.info("ACS %d: prime-age filter %d → %d", year, n_before, len(df))

    df = drop_outlier_wages(df, wage_col="hourly_wage_real")
    df["hourly_wage_real"] = winsorize_wages(df["hourly_wage_real"])
    df["log_hourly_wage_real"] = log_wage(df["hourly_wage_real"])
    df["age_sq"] = df["age"] ** 2

    # Crosswalks
    if "occupation_code" in df.columns:
        occ_num = pd.to_numeric(df["occupation_code"], errors="coerce")
        df["occupation_broad"] = soc_major_to_broad(census_occ_to_soc_major(occ_num))
    if "industry_code" in df.columns:
        ind_num = pd.to_numeric(df["industry_code"], errors="coerce")
        df["industry_broad"] = naics2_to_broad(census_ind_to_naics2(ind_num))
    if "state_fips" in df.columns:
        df["state_abbr"] = state_fips_to_abbr(df["state_fips"])
        df["region"] = state_fips_to_region(df["state_fips"])
    if {"state_fips", "residence_puma"}.issubset(df.columns):
        df = append_puma_cbsa_crosswalk(df)

    # Drop missing outcome
    valid = df["log_hourly_wage_real"].notna() & df["female"].notna()
    df = df[valid].copy()
    logger.info("ACS %d: analysis-ready %d rows", year, len(df))

    df.to_parquet(processed_path, index=False)
    return df


def run_acs_models_single_year(df: pd.DataFrame, year: int, output_dir: Path) -> dict:
    """Run model stack on a single year of ACS data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    from gender_gap.models.descriptive import raw_gap, gap_by_subgroup
    from gender_gap.models.ols import run_sequential_ols, results_to_dataframe

    # Descriptive
    rg = raw_gap(df, outcome="hourly_wage_real", weight="person_weight")
    results["raw_gap"] = rg
    pd.DataFrame([rg]).to_csv(output_dir / "raw_gap.csv", index=False)

    for col in ["education_level", "occupation_broad", "industry_broad",
                 "race_ethnicity", "region"]:
        if col in df.columns:
            sg = gap_by_subgroup(df, col, outcome="hourly_wage_real",
                                 weight="person_weight")
            sg.to_csv(output_dir / f"gap_by_{col}.csv", index=False)

    # Sequential OLS
    logger.info("ACS %d: running OLS M0-M5...", year)
    ols = run_sequential_ols(df, weight_col="person_weight")
    ols_df = results_to_dataframe(ols)
    ols_df.to_csv(output_dir / "ols_sequential.csv", index=False)
    results["ols"] = ols

    # Oaxaca-Blinder
    from gender_gap.models.oaxaca import oaxaca_blinder, oaxaca_summary_table
    try:
        ob = oaxaca_blinder(df, weight_col="person_weight")
        oaxaca_summary_table(ob).to_csv(output_dir / "oaxaca.csv", index=False)
        results["oaxaca"] = ob
    except Exception as e:
        logger.warning("ACS %d Oaxaca failed: %s", year, e)

    # Quantile regression
    from gender_gap.models.quantile import (
        run_quantile_regression, quantile_results_to_dataframe,
        diagnose_distributional_pattern,
    )
    try:
        qr = run_quantile_regression(df, weight_col="person_weight")
        quantile_results_to_dataframe(qr).to_csv(
            output_dir / "quantile_regression.csv", index=False
        )
        results["quantile"] = {
            "results": qr,
            "pattern": diagnose_distributional_pattern(qr),
        }
    except Exception as e:
        logger.warning("ACS %d quantile failed: %s", year, e)

    # Heterogeneity
    from gender_gap.models.heterogeneity import run_full_heterogeneity
    try:
        het = run_full_heterogeneity(df, weight_col="person_weight")
        for dim, hr in het.items():
            hr.subgroup_gaps.to_csv(
                output_dir / f"heterogeneity_{dim}.csv", index=False
            )
        results["heterogeneity"] = het
    except Exception as e:
        logger.warning("ACS %d heterogeneity failed: %s", year, e)

    return results


# ═══════════════════════════════════════════════════════════════════
# CPS ASEC STANDARDIZATION & ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def standardize_cps_asec(df_raw: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Standardize CPS ASEC into analysis-ready format.

    CPS ASEC variable mapping:
      A_SEX: 1=male, 2=female
      A_AGE: age
      A_HGA: education (31-46 scale)
      PRDTRACE: race (1-26)
      PEHSPNON: Hispanic (1=Hispanic, 2=Not)
      A_MJOCC: major occupation code
      A_MJIND: major industry code
      A_CLSWKR: class of worker
      A_USLHRS: usual hours/week
      ERN_VAL: total earnings
      WSAL_VAL: wage/salary income
      A_MARITL: marital status (1-7)
      FOWNU6: own children under 6
      FOWNU18: own children under 18
      GESTFIPS: state FIPS
      MARSUPWT: March supplement weight
    """
    if df_raw is None or len(df_raw) == 0:
        return None

    df = df_raw.copy()

    # Female indicator
    df["female"] = (df["A_SEX"] == 2).astype(int)

    # Age
    df["age"] = df["A_AGE"]
    df["age_sq"] = df["age"] ** 2

    # Employment status from ASEC A_WKSTAT
    df["employed"] = df["A_WKSTAT"].isin([2, 3, 4, 5]).astype(int)
    df["labor_force_status"] = _cps_asec_labor_force_status(df["A_WKSTAT"])

    # Education level from A_HGA
    def map_education(hga):
        if pd.isna(hga):
            return "unknown"
        hga = int(hga)
        if hga <= 38:
            return "less_than_hs"
        elif hga == 39:
            return "hs_diploma"
        elif hga in (40, 41, 42):
            return "some_college"
        elif hga == 43:
            return "bachelors"
        elif hga >= 44:
            return "graduate"
        return "unknown"

    df["education_level"] = df["A_HGA"].apply(map_education)

    # Race/ethnicity
    def map_race(row):
        if pd.notna(row.get("PEHSPNON")) and row["PEHSPNON"] == 1:
            return "hispanic"
        race = row.get("PRDTRACE", 0)
        if pd.isna(race):
            return "other"
        race = int(race)
        if race == 1:
            return "white_non_hispanic"
        elif race == 2:
            return "black_non_hispanic"
        elif race in (4, 5, 6, 7, 8, 9, 10):
            return "asian_non_hispanic"
        return "other"

    df["race_ethnicity"] = df.apply(map_race, axis=1)

    # Marital status
    mar_map = {1: "married", 2: "married", 3: "married",
               4: "widowed", 5: "divorced", 6: "separated", 7: "never_married"}
    df["marital_status"] = df["A_MARITL"].map(mar_map).fillna("unknown")

    # Children
    df["number_children"] = df.get("FOWNU18", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["children_under_5"] = df.get("FOWNU6", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Work variables
    df["usual_hours_week"] = df["A_USLHRS"].fillna(0)
    df["annual_earnings"] = _clean_cps_earnings(df["ERN_VAL"]).fillna(0)
    df["wage_salary_income"] = _clean_cps_earnings(df["WSAL_VAL"]).fillna(0)

    # Derive hourly wage: annual earnings / (50 weeks * usual hours)
    annual_hours = df["usual_hours_week"] * 50
    df["hourly_wage_nominal"] = np.where(
        annual_hours > 0,
        df["annual_earnings"] / annual_hours,
        np.nan,
    )

    # Deflate to base year
    cpi_ratio = CPI_U.get(BASE_YEAR, 292.655) / CPI_U.get(year, CPI_U.get(BASE_YEAR, 292.655))
    df["annual_earnings_real"] = df["annual_earnings"] * cpi_ratio
    df["wage_salary_income_real"] = df["wage_salary_income"] * cpi_ratio
    df["hourly_wage_real"] = df["hourly_wage_nominal"] * cpi_ratio

    # Occupation/industry
    df["occupation_code"] = df.get("A_MJOCC", pd.Series(np.nan, index=df.index))
    df["industry_code"] = df.get("A_MJIND", pd.Series(np.nan, index=df.index))
    df["class_of_worker"] = df.get("A_CLSWKR", pd.Series(np.nan, index=df.index))

    # Geography
    df["state_fips"] = df["GESTFIPS"]

    # Weight
    df["person_weight"] = df["MARSUPWT"].fillna(0) / 100  # CPS weights are in hundredths

    # Survey year
    df["survey_year"] = year
    df["calendar_year"] = year

    # Sample filter: prime-age (25-54), employed, positive earnings
    mask = (
        (df["age"] >= 25) & (df["age"] <= 54) &
        (df["annual_earnings"] > 0) &
        (df["usual_hours_week"] > 0) &
        (df["hourly_wage_real"] > 2) & (df["hourly_wage_real"] < 500) &
        (df["person_weight"] > 0)
    )
    df = df[mask].copy()

    # Log wage
    df["log_hourly_wage_real"] = np.log(df["hourly_wage_real"])

    logger.info("CPS ASEC %d: %d analysis-ready rows", year, len(df))
    return df


def _cps_asec_labor_force_status(a_wkstat: pd.Series) -> pd.Series:
    """Map CPS ASEC A_WKSTAT into coarse labor-force groups.

    Census labels:
      2-5 employed, 6-7 unemployed, 1 armed forces/children, 0 NIU.
    """
    result = pd.Series("not_in_labor_force", index=a_wkstat.index)
    result[a_wkstat.isin([2, 3, 4, 5])] = "employed"
    result[a_wkstat.isin([6, 7])] = "unemployed"
    return result


def prepare_cps_asec_selection_sample(df_raw: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Prepare a prime-age CPS ASEC sample for selection robustness checks."""
    if df_raw is None or len(df_raw) == 0:
        return None

    df = df_raw.copy()
    df["female"] = (df["A_SEX"] == 2).astype(int)
    df["age"] = pd.to_numeric(df["A_AGE"], errors="coerce")
    df["age_sq"] = df["age"] ** 2
    df["employed"] = df["A_WKSTAT"].isin([2, 3, 4, 5]).astype(int)
    df["labor_force_status"] = _cps_asec_labor_force_status(df["A_WKSTAT"])

    def map_education(hga):
        if pd.isna(hga):
            return "unknown"
        hga = int(hga)
        if hga <= 38:
            return "less_than_hs"
        if hga == 39:
            return "hs_diploma"
        if hga in (40, 41, 42):
            return "some_college"
        if hga == 43:
            return "bachelors"
        if hga >= 44:
            return "graduate"
        return "unknown"

    def map_race(row):
        if pd.notna(row.get("PEHSPNON")) and row["PEHSPNON"] == 1:
            return "hispanic"
        race = row.get("PRDTRACE", 0)
        if pd.isna(race):
            return "other"
        race = int(race)
        if race == 1:
            return "white_non_hispanic"
        if race == 2:
            return "black_non_hispanic"
        if race in (4, 5, 6, 7, 8, 9, 10):
            return "asian_non_hispanic"
        return "other"

    mar_map = {
        1: "married", 2: "married", 3: "married",
        4: "widowed", 5: "divorced", 6: "separated", 7: "never_married",
    }

    df["education_level"] = df["A_HGA"].apply(map_education)
    df["race_ethnicity"] = df.apply(map_race, axis=1)
    df["marital_status"] = df["A_MARITL"].map(mar_map).fillna("unknown")
    df["number_children"] = df.get("FOWNU18", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["children_under_5"] = df.get("FOWNU6", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["state_fips"] = df["GESTFIPS"]
    df["usual_hours_week"] = pd.to_numeric(df["A_USLHRS"], errors="coerce").fillna(0)
    df["annual_earnings"] = _clean_cps_earnings(df["ERN_VAL"])
    df["wage_salary_income"] = _clean_cps_earnings(df["WSAL_VAL"])
    df["person_weight"] = pd.to_numeric(df["MARSUPWT"], errors="coerce").fillna(0) / 100

    annual_hours = df["usual_hours_week"] * 50
    df["hourly_wage_nominal"] = np.where(
        annual_hours > 0,
        df["annual_earnings"] / annual_hours,
        np.nan,
    )
    cpi_ratio = CPI_U.get(BASE_YEAR, 292.655) / CPI_U.get(year, CPI_U.get(BASE_YEAR, 292.655))
    df["annual_earnings_real"] = df["annual_earnings"] * cpi_ratio
    df["hourly_wage_real"] = df["hourly_wage_nominal"] * cpi_ratio
    df["survey_year"] = year
    df["calendar_year"] = year

    mask = (
        df["age"].between(25, 54)
        & (df["person_weight"] > 0)
        & (df["A_WKSTAT"] != 1)
    )
    df = df.loc[mask].copy()
    logger.info("CPS ASEC %d: %d selection-sample rows", year, len(df))
    return df


def run_cps_models(
    df: pd.DataFrame,
    year: int,
    output_dir: Path,
    selection_df: pd.DataFrame | None = None,
) -> dict:
    """Run models on CPS ASEC data (lighter stack due to smaller sample)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    from gender_gap.models.descriptive import raw_gap
    from gender_gap.models.ols import run_sequential_ols, results_to_dataframe, BLOCK_DEFINITIONS

    # Descriptive
    rg = raw_gap(df, outcome="hourly_wage_real", weight="person_weight")
    results["raw_gap"] = rg
    pd.DataFrame([rg]).to_csv(output_dir / "raw_gap.csv", index=False)

    # CPS has different variable availability — build adapted blocks
    cps_blocks = {}
    available = set(df.columns)

    cps_blocks["M0"] = ["female"]
    if {"age", "age_sq"}.issubset(available):
        cps_blocks["M1"] = ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"]
    if "state_fips" in available:
        cps_blocks["M2"] = cps_blocks.get("M1", ["female"]) + ["C(state_fips)"]
    if "occupation_code" in available:
        m3_controls = cps_blocks.get("M2", cps_blocks.get("M1", ["female"]))
        cps_blocks["M3"] = m3_controls + ["C(occupation_code)", "C(industry_code)"]

    # Add family controls
    last_block = max(cps_blocks.keys()) if cps_blocks else "M0"
    if "marital_status" in available:
        cps_blocks["M_full"] = cps_blocks[last_block] + [
            "C(marital_status)", "number_children", "children_under_5"
        ]

    logger.info("CPS %d: running OLS with %d blocks...", year, len(cps_blocks))
    ols = run_sequential_ols(df, weight_col="person_weight", blocks=cps_blocks)
    results_to_dataframe(ols).to_csv(output_dir / "ols_sequential.csv", index=False)
    results["ols"] = ols

    if selection_df is not None and len(selection_df) > 0:
        from gender_gap.models.selection import run_selection_robustness

        selection = run_selection_robustness(selection_df)
        selection.to_csv(output_dir / "selection_robustness.csv", index=False)
        results["selection"] = selection

    return results


# ═══════════════════════════════════════════════════════════════════
# ATUS STANDARDIZATION & ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def standardize_atus_data(atus_data: dict) -> pd.DataFrame | None:
    """Standardize ATUS activity summary data."""
    if "activity_summary" not in atus_data:
        logger.warning("ATUS: activity_summary not available")
        return None

    from gender_gap.standardize.atus_standardize import standardize_atus_summary

    try:
        df = standardize_atus_summary(atus_data["activity_summary"])
        logger.info("ATUS: standardized %d person-days", len(df))
        return df
    except Exception as e:
        logger.error("ATUS standardization failed: %s", e)
        return None


def run_atus_analysis(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run time-use mechanism analysis on ATUS data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Time use by gender
    time_vars = [
        "minutes_paid_work_diary", "minutes_housework",
        "minutes_childcare", "minutes_commute_related_travel",
    ]
    available_vars = [v for v in time_vars if v in df.columns]

    if not available_vars:
        logger.warning("ATUS: no time-use variables found")
        return results

    # Gender means
    weight_col = "person_weight" if "person_weight" in df.columns else None

    rows = []
    for var in available_vars:
        if weight_col and weight_col in df.columns:
            male_mean = np.average(
                df.loc[df["female"] == 0, var].dropna(),
                weights=df.loc[df["female"] == 0, weight_col].iloc[:len(df.loc[df["female"] == 0, var].dropna())]
            ) if len(df.loc[df["female"] == 0, var].dropna()) > 0 else np.nan
            female_mean = np.average(
                df.loc[df["female"] == 1, var].dropna(),
                weights=df.loc[df["female"] == 1, weight_col].iloc[:len(df.loc[df["female"] == 1, var].dropna())]
            ) if len(df.loc[df["female"] == 1, var].dropna()) > 0 else np.nan
        else:
            male_mean = df.loc[df["female"] == 0, var].mean()
            female_mean = df.loc[df["female"] == 1, var].mean()

        rows.append({
            "activity": var,
            "male_mean_minutes": male_mean,
            "female_mean_minutes": female_mean,
            "gap_minutes": female_mean - male_mean,
        })

    time_use_df = pd.DataFrame(rows)
    time_use_df.to_csv(output_dir / "time_use_by_gender.csv", index=False)
    results["time_use"] = time_use_df
    logger.info("ATUS: time-use analysis complete")

    return results


# ═══════════════════════════════════════════════════════════════════
# TREND ANALYSIS (Cross-Year)
# ═══════════════════════════════════════════════════════════════════

def compile_trend_results(results_by_year: dict, source: str, output_dir: Path):
    """Compile multi-year results into trend tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw gap trend
    raw_trend = []
    for year, res in sorted(results_by_year.items()):
        if "raw_gap" in res:
            rg = res["raw_gap"]
            raw_trend.append({
                "year": year,
                "source": source,
                "male_mean": rg["male_mean"],
                "female_mean": rg["female_mean"],
                "gap_dollars": rg["gap_dollars"],
                "gap_pct": rg["gap_pct"],
                "n_male": rg["n_male"],
                "n_female": rg["n_female"],
            })

    if raw_trend:
        pd.DataFrame(raw_trend).to_csv(
            output_dir / f"{source}_raw_gap_trend.csv", index=False
        )

    # OLS coefficient trend
    ols_trend = []
    for year, res in sorted(results_by_year.items()):
        if "ols" in res:
            for r in res["ols"]:
                ols_trend.append({
                    "year": year,
                    "source": source,
                    "model": r.model_name,
                    "female_coef": r.female_coef,
                    "female_se": r.female_se,
                    "female_pvalue": r.female_pvalue,
                    "r_squared": r.r_squared,
                    "n_obs": r.n_obs,
                    "pct_gap": (np.exp(r.female_coef) - 1) * 100,
                })

    if ols_trend:
        pd.DataFrame(ols_trend).to_csv(
            output_dir / f"{source}_ols_trend.csv", index=False
        )

    # Oaxaca trend
    oaxaca_trend = []
    for year, res in sorted(results_by_year.items()):
        if "oaxaca" in res:
            ob = res["oaxaca"]
            oaxaca_trend.append({
                "year": year,
                "source": source,
                "total_gap": ob.total_gap,
                "explained": ob.explained,
                "unexplained": ob.unexplained,
                "explained_pct": ob.explained_pct,
                "unexplained_pct": ob.unexplained_pct,
            })

    if oaxaca_trend:
        pd.DataFrame(oaxaca_trend).to_csv(
            output_dir / f"{source}_oaxaca_trend.csv", index=False
        )

    logger.info("%s trend: %d years compiled", source, len(results_by_year))


def _acs_year_outputs_complete(output_dir: Path) -> bool:
    """Return True when the expensive ACS model outputs already exist."""
    expected = [
        "raw_gap.csv",
        "ols_sequential.csv",
        "oaxaca.csv",
        "quantile_regression.csv",
        "heterogeneity_education_level.csv",
        "heterogeneity_race_ethnicity.csv",
        "heterogeneity_marital_status.csv",
        "heterogeneity_occupation_broad.csv",
        "heterogeneity_industry_broad.csv",
        "heterogeneity_work_from_home.csv",
        "heterogeneity_state_fips.csv",
    ]
    return all((output_dir / name).exists() for name in expected)


def _cps_year_outputs_complete(output_dir: Path) -> bool:
    """Return True when the CPS model outputs already exist."""
    expected = ["raw_gap.csv", "ols_sequential.csv"]
    return all((output_dir / name).exists() for name in expected)


def _cps_selection_output_complete(output_dir: Path) -> bool:
    """Return True when the CPS selection robustness output exists."""
    return (output_dir / "selection_robustness.csv").exists()


def _pooled_acs_outputs_complete(output_dir: Path) -> bool:
    """Return True when pooled ACS outputs already exist."""
    expected = ["raw_gap_pooled.csv", "ols_pooled.csv", "oaxaca_pooled.csv"]
    return all((output_dir / name).exists() for name in expected)


def _load_saved_acs_results(output_dir: Path) -> dict:
    """Load previously written ACS result summaries for trend compilation."""
    from gender_gap.models.ols import OLSResult

    results: dict = {}

    raw_gap_path = output_dir / "raw_gap.csv"
    if raw_gap_path.exists():
        results["raw_gap"] = pd.read_csv(raw_gap_path).iloc[0].to_dict()

    ols_path = output_dir / "ols_sequential.csv"
    if ols_path.exists():
        ols_df = pd.read_csv(ols_path)
        results["ols"] = [
            OLSResult(
                model_name=row["model"],
                female_coef=row["female_coef"],
                female_se=row["female_se"],
                female_pvalue=row["female_pvalue"],
                r_squared=row["r_squared"],
                n_obs=int(row["n_obs"]),
                controls=[],
            )
            for _, row in ols_df.iterrows()
        ]

    oaxaca_path = output_dir / "oaxaca.csv"
    if oaxaca_path.exists():
        oaxaca_df = pd.read_csv(oaxaca_path)
        value_map = dict(zip(oaxaca_df["component"], oaxaca_df["value"], strict=False))
        pct_map = dict(zip(oaxaca_df["component"], oaxaca_df["pct"], strict=False))
        results["oaxaca"] = SimpleNamespace(
            total_gap=value_map.get("Total gap", np.nan),
            explained=value_map.get("Explained (endowments)", np.nan),
            unexplained=value_map.get("Unexplained (coefficients)", np.nan),
            explained_pct=pct_map.get("Explained (endowments)", np.nan),
            unexplained_pct=pct_map.get("Unexplained (coefficients)", np.nan),
        )

    return results


def _load_saved_cps_results(output_dir: Path) -> dict:
    """Load previously written CPS result summaries for trend compilation."""
    from gender_gap.models.ols import OLSResult

    results: dict = {}

    raw_gap_path = output_dir / "raw_gap.csv"
    if raw_gap_path.exists():
        results["raw_gap"] = pd.read_csv(raw_gap_path).iloc[0].to_dict()

    ols_path = output_dir / "ols_sequential.csv"
    if ols_path.exists():
        ols_df = pd.read_csv(ols_path)
        results["ols"] = [
            OLSResult(
                model_name=row["model"],
                female_coef=row["female_coef"],
                female_se=row["female_se"],
                female_pvalue=row["female_pvalue"],
                r_squared=row["r_squared"],
                n_obs=int(row["n_obs"]),
                controls=[],
            )
            for _, row in ols_df.iterrows()
        ]

    return results


def _pooled_job_terms(df: pd.DataFrame) -> tuple[str, str]:
    """Choose tractable pooled occupation/industry controls.

    Detailed occupation and industry codes across the pooled ACS panel create a
    very wide design matrix over ~7M rows. Prefer broad categories when they
    are already available from the year-level standardization step.
    """
    occ_term = "C(occupation_broad)" if "occupation_broad" in df.columns else "C(occupation_code)"
    ind_term = "C(industry_broad)" if "industry_broad" in df.columns else "C(industry_code)"
    return occ_term, ind_term


# ═══════════════════════════════════════════════════════════════════
# POOLED PANEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def run_pooled_acs_analysis(years_data: dict[int, pd.DataFrame], output_dir: Path):
    """Pool all ACS years and run models with year fixed effects."""
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for year, df in sorted(years_data.items()):
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        logger.warning("No ACS data for pooled analysis")
        return

    pooled = pd.concat(frames, ignore_index=True)
    logger.info("Pooled ACS: %d rows across %d years", len(pooled), len(frames))

    if _pooled_acs_outputs_complete(output_dir):
        logger.info("Pooled ACS: using existing outputs at %s", output_dir)
        return

    from gender_gap.models.ols import run_sequential_ols, results_to_dataframe
    from gender_gap.models.descriptive import raw_gap
    from gender_gap.models.oaxaca import oaxaca_blinder, oaxaca_summary_table

    rg = raw_gap(pooled, outcome="hourly_wage_real", weight="person_weight")
    pd.DataFrame([rg]).to_csv(output_dir / "raw_gap_pooled.csv", index=False)

    occ_term, ind_term = _pooled_job_terms(pooled)
    logger.info(
        "Pooled ACS: using %s and %s for job controls",
        occ_term,
        ind_term,
    )

    # Pooled OLS with year FE
    pooled_blocks = {
        "P0": ["female"],
        "P1": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)"],
        "P2": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
                "C(state_fips)", "C(calendar_year)"],
        "P3": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
                "C(state_fips)", "C(calendar_year)",
                occ_term, ind_term],
        "P4": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
                "C(state_fips)", "C(calendar_year)",
                occ_term, ind_term,
                "usual_hours_week", "work_from_home", "commute_minutes_one_way"],
        "P5": ["female", "age", "age_sq", "C(race_ethnicity)", "C(education_level)",
                "C(state_fips)", "C(calendar_year)",
                occ_term, ind_term,
                "usual_hours_week", "work_from_home", "commute_minutes_one_way",
                "C(marital_status)", "number_children", "children_under_5"],
    }

    logger.info("Running pooled OLS P0-P5 (%d observations)...", len(pooled))
    ols = run_sequential_ols(pooled, weight_col="person_weight", blocks=pooled_blocks)
    results_to_dataframe(ols).to_csv(output_dir / "ols_pooled.csv", index=False)

    # Oaxaca on pooled data
    try:
        ob = oaxaca_blinder(pooled, weight_col="person_weight")
        oaxaca_summary_table(ob).to_csv(output_dir / "oaxaca_pooled.csv", index=False)
    except Exception as e:
        logger.warning("Pooled Oaxaca failed: %s", e)

    logger.info("Pooled ACS analysis complete")


# ═══════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    # ACS 2020 1-year was not released (COVID collection suspension)
    ACS_YEARS = [y for y in range(2015, 2024) if y != 2020]
    CPS_YEARS = list(range(2015, 2024))

    logger.info("=" * 75)
    logger.info("RUNNING ALL ANALYSES")
    logger.info("=" * 75)

    # ── 1. ACS — Year-by-year ──
    logger.info("\n" + "═" * 60)
    logger.info("1. ACS PUMS YEAR-BY-YEAR ANALYSIS")
    logger.info("═" * 60)

    acs_processed = {}
    acs_results = {}

    for year in ACS_YEARS:
        try:
            df = standardize_acs_year(year)
            if df is not None and len(df) > 0:
                acs_processed[year] = df
                year_dir = RESULTS_DIR / "acs" / str(year)
                if _acs_year_outputs_complete(year_dir):
                    logger.info("ACS %d: using existing outputs at %s", year, year_dir)
                    acs_results[year] = _load_saved_acs_results(year_dir)
                else:
                    acs_results[year] = run_acs_models_single_year(df, year, year_dir)
                logger.info("ACS %d: complete — N=%d", year, len(df))
        except Exception as e:
            logger.error("ACS %d analysis failed: %s", year, e, exc_info=True)

    # Compile ACS trends
    if acs_results:
        compile_trend_results(acs_results, "acs", RESULTS_DIR / "trends")

    # ── 2. ACS Pooled Panel ──
    logger.info("\n" + "═" * 60)
    logger.info("2. ACS POOLED PANEL (all years combined)")
    logger.info("═" * 60)

    if len(acs_processed) > 1:
        run_pooled_acs_analysis(acs_processed, RESULTS_DIR / "acs_pooled")

    # ── 3. CPS ASEC — Year-by-year ──
    logger.info("\n" + "═" * 60)
    logger.info("3. CPS ASEC YEAR-BY-YEAR ANALYSIS")
    logger.info("═" * 60)

    cps_results = {}
    for year in CPS_YEARS:
        raw_path = DATA_RAW / "cps" / f"cps_asec_{year}_api.parquet"
        if not raw_path.exists():
            logger.warning("CPS %d: no raw data", year)
            continue
        try:
            df_raw = pd.read_parquet(raw_path)
            for col in CPS_NUMERIC_COLS:
                if col in df_raw.columns:
                    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

            df = standardize_cps_asec(df_raw, year)
            selection_df = prepare_cps_asec_selection_sample(df_raw, year)
            if df is not None and len(df) > 0:
                df.to_parquet(
                    DATA_PROCESSED / f"cps_asec_{year}_analysis_ready.parquet",
                    index=False,
                )
                year_dir = RESULTS_DIR / "cps" / str(year)
                if _cps_year_outputs_complete(year_dir):
                    logger.info("CPS %d: using existing outputs at %s", year, year_dir)
                    cps_results[year] = _load_saved_cps_results(year_dir)
                    if (
                        selection_df is not None and len(selection_df) > 0
                        and not _cps_selection_output_complete(year_dir)
                    ):
                        from gender_gap.models.selection import run_selection_robustness

                        selection = run_selection_robustness(selection_df)
                        selection.to_csv(year_dir / "selection_robustness.csv", index=False)
                        cps_results[year]["selection"] = selection
                else:
                    cps_results[year] = run_cps_models(
                        df,
                        year,
                        year_dir,
                        selection_df=selection_df,
                    )
                logger.info("CPS %d: complete — N=%d", year, len(df))
        except Exception as e:
            logger.error("CPS %d analysis failed: %s", year, e, exc_info=True)

    if cps_results:
        compile_trend_results(cps_results, "cps_asec", RESULTS_DIR / "trends")

    # ── 4. ATUS ──
    logger.info("\n" + "═" * 60)
    logger.info("4. ATUS TIME-USE ANALYSIS")
    logger.info("═" * 60)

    atus_dir = DATA_RAW / "atus"
    atus_data = {}
    for name in ["activity_summary", "respondent", "cps_link"]:
        path = atus_dir / f"atus_{name}.parquet"
        if path.exists():
            atus_data[name] = pd.read_parquet(path)

    if atus_data:
        atus_df = standardize_atus_data(atus_data)
        if atus_df is not None:
            atus_df.to_parquet(
                DATA_PROCESSED / "atus_analysis_ready.parquet", index=False
            )
            run_atus_analysis(atus_df, RESULTS_DIR / "atus")

    # ── 5. Print Summary ──
    elapsed = time.time() - start

    logger.info("\n" + "=" * 75)
    logger.info("ALL ANALYSES COMPLETE (%.1f minutes)", elapsed / 60)
    logger.info("=" * 75)

    # Summary table
    print("\n" + "=" * 80)
    print("COMPLETE RESULTS SUMMARY")
    print("=" * 80)

    # ACS trend
    if acs_results:
        print("\nACS RAW GAP TREND:")
        print(f"  {'Year':<8} {'Male':>10} {'Female':>10} {'Gap%':>8} {'N':>12}")
        print(f"  {'-' * 52}")
        for year in sorted(acs_results):
            rg = acs_results[year].get("raw_gap", {})
            if rg:
                print(f"  {year:<8} ${rg['male_mean']:>9.2f} ${rg['female_mean']:>9.2f} "
                      f"{rg['gap_pct']:>7.1f}% {rg['n_male'] + rg['n_female']:>11,}")

        print("\nACS ADJUSTED GAP TREND (M5 — full controls):")
        print(f"  {'Year':<8} {'Female coef':>12} {'% gap':>8} {'R²':>8} {'N':>10}")
        print(f"  {'-' * 50}")
        for year in sorted(acs_results):
            ols = acs_results[year].get("ols", [])
            m5 = [r for r in ols if r.model_name == "M5"]
            if m5:
                r = m5[0]
                pct = (np.exp(r.female_coef) - 1) * 100
                print(f"  {year:<8} {r.female_coef:>12.4f} {pct:>7.1f}% "
                      f"{r.r_squared:>8.4f} {r.n_obs:>10,}")

    # CPS trend
    if cps_results:
        print("\nCPS ASEC RAW GAP TREND:")
        print(f"  {'Year':<8} {'Male':>10} {'Female':>10} {'Gap%':>8} {'N':>12}")
        print(f"  {'-' * 52}")
        for year in sorted(cps_results):
            rg = cps_results[year].get("raw_gap", {})
            if rg:
                print(f"  {year:<8} ${rg['male_mean']:>9.2f} ${rg['female_mean']:>9.2f} "
                      f"{rg['gap_pct']:>7.1f}% {rg['n_male'] + rg['n_female']:>11,}")

    print("\n" + "=" * 80)


# CPS numeric cols (reuse from download script)
CPS_NUMERIC_COLS = [
    "A_SEX", "A_AGE", "A_HGA", "PRDTRACE", "PEHSPNON",
    "A_MJOCC", "A_MJIND", "A_CLSWKR", "A_USLHRS", "A_HRS1",
    "ERN_VAL", "WSAL_VAL", "WS_VAL", "A_WKSTAT",
    "A_MARITL", "FOWNU6", "FOWNU18",
    "GESTFIPS", "MARSUPWT",
]


if __name__ == "__main__":
    main()
