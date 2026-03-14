#!/usr/bin/env python3
"""Full pipeline: download ACS PUMS via Census API, standardize, run all models.

Usage:
    python scripts/run_full_analysis.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")

# All 50 states + DC (FIPS codes)
ALL_STATES = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56",
]

# ACS PUMS variables we need
ACS_VARS = [
    "SERIALNO", "SPORDER", "SEX", "AGEP", "SCHL", "HISP", "RAC1P",
    "WAGP", "PERNP", "WKHP", "WKWN", "COW", "OCCP", "INDP",
    "JWTRNS", "JWMNP", "ST", "PUMA", "PWGTP", "MAR", "ADJINC",
]

# CPI-U annual averages for deflation (BLS series CUUR0000SA0)
CPI_U = {
    2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107,
    2019: 255.657, 2020: 258.811, 2021: 270.970, 2022: 292.655,
    2023: 304.702, 2024: 312.332,
}


def download_acs_pums(year: int) -> pd.DataFrame:
    """Download ACS PUMS microdata via Census API."""
    cache_path = DATA_RAW / "acs" / f"acs_pums_{year}_api.parquet"
    if cache_path.exists():
        logger.info("Loading cached ACS %d from %s", year, cache_path)
        df = pd.read_parquet(cache_path)
        # Ensure numeric columns are numeric (API returns strings)
        for col in ["SEX", "AGEP", "WAGP", "PERNP", "WKHP", "WKWN", "COW",
                     "JWTRNS", "JWMNP", "ST", "PWGTP", "MAR", "ADJINC",
                     "HISP", "SCHL", "RAC1P", "SPORDER", "OCCP", "INDP"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Census API ADJINC fix (see below)
        if "ADJINC" in df.columns and df["ADJINC"].max() < 100:
            df["ADJINC"] = df["ADJINC"] * 1_000_000
        return df

    logger.info("Downloading ACS PUMS %d via Census API...", year)
    vars_str = ",".join(ACS_VARS)
    all_rows = []

    for i, st in enumerate(ALL_STATES):
        url = (
            f"https://api.census.gov/data/{year}/acs/acs1/pums"
            f"?get={vars_str}&for=state:{st}&key={CENSUS_API_KEY}"
        )
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                data = r.json()
                if i == 0:
                    header = data[0]
                rows = data[1:]
                all_rows.extend(rows)
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning(
                        "State %s attempt %d failed: %s, retrying...",
                        st, attempt + 1, e,
                    )
                    time.sleep(2)
                else:
                    logger.error("State %s failed after 3 attempts: %s", st, e)

        if (i + 1) % 10 == 0:
            logger.info(
                "  %d/%d states, %d rows so far", i + 1, len(ALL_STATES), len(all_rows)
            )

    logger.info("Downloaded %d total records for ACS %d", len(all_rows), year)

    df = pd.DataFrame(all_rows, columns=header)

    # Convert ALL numeric columns (Census API returns everything as strings)
    numeric_cols = [
        "SEX", "AGEP", "WAGP", "PERNP", "WKHP", "WKWN", "COW",
        "JWTRNS", "JWMNP", "ST", "PWGTP", "MAR", "ADJINC", "HISP",
        "SCHL", "RAC1P", "SPORDER",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # OCCP and INDP can be numeric or alphanumeric — keep as string
    # but try numeric for standardizer compatibility
    for col in ["OCCP", "INDP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Census API returns ADJINC as a decimal (1.042311) but the ACS
    # standardizer expects the raw integer format (1042311) and divides
    # by 1,000,000 internally. Convert API format to raw format.
    if "ADJINC" in df.columns and df["ADJINC"].max() < 100:
        df["ADJINC"] = df["ADJINC"] * 1_000_000

    # Cache to parquet
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached ACS %d to %s", year, cache_path)

    return df


def standardize_and_build_features(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Standardize ACS PUMS and build analysis features."""
    from gender_gap.standardize.acs_standardize import standardize_acs
    from gender_gap.features.earnings import winsorize_wages, log_wage
    from gender_gap.features.sample_filters import (
        filter_prime_age_wage_salary, drop_outlier_wages,
    )
    from gender_gap.features.family import parenthood_category
    from gender_gap.features.commute import commute_bin
    from gender_gap.crosswalks.occupation_crosswalks import (
        census_occ_to_soc_major, soc_major_to_label, soc_major_to_broad,
    )
    from gender_gap.crosswalks.industry_crosswalks import (
        census_ind_to_naics2, naics2_to_label, naics2_to_broad,
    )
    from gender_gap.crosswalks.geography_crosswalks import (
        append_puma_cbsa_crosswalk, state_fips_to_abbr, state_fips_to_region,
    )

    logger.info("Standardizing ACS %d (%d rows)...", year, len(df_raw))
    df = standardize_acs(df_raw, survey_year=year)

    # Ensure hourly_wage_real is numeric
    df["hourly_wage_real"] = pd.to_numeric(df["hourly_wage_real"], errors="coerce")

    # Sample filter: prime-age wage/salary workers
    n_before = len(df)
    df = filter_prime_age_wage_salary(df)
    logger.info("Prime-age filter: %d → %d", n_before, len(df))

    # Drop outlier wages
    df = drop_outlier_wages(df, wage_col="hourly_wage_real")
    logger.info("After outlier drop: %d", len(df))

    # Winsorize
    df["hourly_wage_real"] = winsorize_wages(df["hourly_wage_real"])

    # Log wage
    df["log_hourly_wage_real"] = log_wage(df["hourly_wage_real"])

    # Age squared
    df["age_sq"] = df["age"] ** 2

    # Occupation crosswalks
    if "occupation_code" in df.columns:
        occ_num = pd.to_numeric(df["occupation_code"], errors="coerce")
        soc2 = census_occ_to_soc_major(occ_num)
        df["occupation_soc2"] = soc2
        df["occupation_label"] = soc_major_to_label(soc2)
        df["occupation_broad"] = soc_major_to_broad(soc2)

    # Industry crosswalks
    if "industry_code" in df.columns:
        ind_num = pd.to_numeric(df["industry_code"], errors="coerce")
        naics2 = census_ind_to_naics2(ind_num)
        df["industry_naics2"] = naics2
        df["industry_label"] = naics2_to_label(naics2)
        df["industry_broad"] = naics2_to_broad(naics2)

    # Geography
    if "state_fips" in df.columns:
        df["state_abbr"] = state_fips_to_abbr(df["state_fips"])
        df["region"] = state_fips_to_region(df["state_fips"])
    if {"state_fips", "residence_puma"}.issubset(df.columns):
        df = append_puma_cbsa_crosswalk(df)

    # Commute
    if "commute_minutes_one_way" in df.columns:
        df["commute_bin"] = commute_bin(df["commute_minutes_one_way"])

    # Drop rows with missing outcome
    valid = df["log_hourly_wage_real"].notna() & df["female"].notna()
    df = df[valid].copy()
    logger.info("Analysis-ready: %d rows", len(df))

    return df


def run_all_models(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run the complete model stack."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── 1. Descriptive ──
    from gender_gap.models.descriptive import raw_gap, gap_by_subgroup
    logger.info("Running descriptive models...")

    rg = raw_gap(df, outcome="hourly_wage_real", weight="person_weight")
    results["raw_gap"] = rg
    pd.DataFrame([rg]).to_csv(output_dir / "raw_gap.csv", index=False)

    # Subgroup gaps
    for col in ["education_level", "occupation_broad", "industry_broad",
                 "race_ethnicity", "region"]:
        if col in df.columns:
            sg = gap_by_subgroup(df, col, outcome="hourly_wage_real",
                                 weight="person_weight")
            sg.to_csv(output_dir / f"gap_by_{col}.csv", index=False)

    # ── 2. Sequential OLS (M0-M5) ──
    from gender_gap.models.ols import (
        run_sequential_ols, BLOCK_DEFINITIONS, results_to_dataframe,
    )
    logger.info("Running sequential OLS (M0-M5)...")
    ols = run_sequential_ols(df, weight_col="person_weight")
    ols_df = results_to_dataframe(ols)
    ols_df.to_csv(output_dir / "ols_sequential.csv", index=False)
    results["ols"] = ols

    # ── 3. Oaxaca-Blinder ──
    from gender_gap.models.oaxaca import oaxaca_blinder, oaxaca_summary_table
    logger.info("Running Oaxaca-Blinder decomposition...")
    try:
        ob = oaxaca_blinder(df, weight_col="person_weight")
        oaxaca_summary_table(ob).to_csv(
            output_dir / "oaxaca.csv", index=False
        )
        results["oaxaca"] = ob
    except Exception as e:
        logger.warning("Oaxaca-Blinder failed: %s", e)

    # ── 4. Quantile regression ──
    from gender_gap.models.quantile import (
        run_quantile_regression, quantile_results_to_dataframe,
        diagnose_distributional_pattern,
    )
    logger.info("Running quantile regression...")
    qr = run_quantile_regression(df, weight_col="person_weight")
    quantile_results_to_dataframe(qr).to_csv(
        output_dir / "quantile_regression.csv", index=False
    )
    pattern = diagnose_distributional_pattern(qr)
    results["quantile"] = {"results": qr, "pattern": pattern}

    # ── 5. Heterogeneity ──
    from gender_gap.models.heterogeneity import run_full_heterogeneity
    logger.info("Running heterogeneity analysis...")
    het = run_full_heterogeneity(df, weight_col="person_weight")
    for dim, hr in het.items():
        hr.subgroup_gaps.to_csv(
            output_dir / f"heterogeneity_{dim}.csv", index=False
        )
    results["heterogeneity"] = het

    # ── 6. Elastic Net ──
    from gender_gap.models.elastic_net import run_elastic_net
    logger.info("Running Elastic Net...")
    try:
        en = run_elastic_net(df, weight_col="person_weight")
        en.top_interactions.to_csv(
            output_dir / "elastic_net_interactions.csv", index=False
        )
        results["elastic_net"] = en
    except Exception as e:
        logger.warning("Elastic Net failed: %s", e)

    # ── 7. DoubleML ──
    from gender_gap.models.dml import run_dml
    logger.info("Running DoubleML...")
    try:
        dml_result = run_dml(df, weight_col="person_weight")
        pd.DataFrame([{
            "treatment_effect": dml_result.treatment_effect,
            "std_error": dml_result.std_error,
            "ci_lower": dml_result.ci_lower,
            "ci_upper": dml_result.ci_upper,
            "pvalue": dml_result.pvalue,
        }]).to_csv(output_dir / "dml.csv", index=False)
        results["dml"] = dml_result
    except Exception as e:
        logger.warning("DoubleML failed: %s", e)

    return results


def print_results(results: dict) -> None:
    """Print formatted results to stdout."""
    print("\n" + "=" * 75)
    print("GENDER EARNINGS GAP ANALYSIS — ACS PUMS RESULTS")
    print("=" * 75)

    # Raw gap
    rg = results["raw_gap"]
    print("\n1. RAW GAP (unadjusted)")
    print("-" * 50)
    print(f"   Male mean hourly wage:   ${rg['male_mean']:.2f}")
    print(f"   Female mean hourly wage: ${rg['female_mean']:.2f}")
    print(f"   Raw gap:                 ${rg['gap_dollars']:.2f} "
          f"({rg['gap_pct']:.1f}%)")
    print(f"   N male: {rg['n_male']:,}  |  N female: {rg['n_female']:,}")

    # OLS
    if "ols" in results:
        ols = results["ols"]
        print("\n2. SEQUENTIAL OLS — Female coefficient (log hourly wage)")
        print("-" * 70)
        print(f"   {'Model':<8} {'Female coef':>12} {'SE':>8} "
              f"{'p-value':>10} {'R²':>8} {'N':>8}  {'% gap':>8}")
        print(f"   {'-' * 62}")
        for r in ols:
            pct = f"({(np.exp(r.female_coef) - 1) * 100:.1f}%)"
            sig = "***" if r.female_pvalue < 0.001 else ""
            print(f"   {r.model_name:<8} {r.female_coef:>12.4f} "
                  f"{r.female_se:>8.4f} {r.female_pvalue:>10.4f} "
                  f"{r.r_squared:>8.4f} {r.n_obs:>8,}  {pct:>8} {sig}")

    # Oaxaca
    if "oaxaca" in results:
        ob = results["oaxaca"]
        print("\n3. OAXACA-BLINDER DECOMPOSITION")
        print("-" * 50)
        print(f"   Total gap (log wage):    {ob.total_gap:.4f}")
        print(f"   Explained (endowments):  {ob.explained:.4f} "
              f"({ob.explained_pct:.1f}%)")
        print(f"   Unexplained (coefs):     {ob.unexplained:.4f} "
              f"({ob.unexplained_pct:.1f}%)")

    # Quantile
    if "quantile" in results:
        qr = results["quantile"]["results"]
        pattern = results["quantile"]["pattern"]
        print(f"\n4. QUANTILE REGRESSION (pattern: {pattern})")
        print("-" * 60)
        print(f"   {'Quantile':>10} {'Female coef':>12} {'SE':>8} "
              f"{'p-value':>10} {'% gap':>8}")
        print(f"   {'-' * 50}")
        for r in qr:
            pct = f"{(np.exp(r.female_coef) - 1) * 100:.1f}%"
            print(f"   {r.quantile:>10.0%} {r.female_coef:>12.4f} "
                  f"{r.female_se:>8.4f} {r.female_pvalue:>10.4f} "
                  f"{pct:>8}")

    # Heterogeneity
    if "heterogeneity" in results:
        print("\n5. HETEROGENEITY — Gap by subgroup")
        print("-" * 75)
        for dim, hr in results["heterogeneity"].items():
            if len(hr.subgroup_gaps) == 0:
                continue
            print(f"\n   By {dim}:")
            sorted_gaps = hr.subgroup_gaps.sort_values("gap")
            for _, row in sorted_gaps.iterrows():
                pct = f"({(np.exp(row['gap']) - 1) * 100:.1f}%)"
                ci = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                print(f"     {str(row['group']):<30} {row['gap']:>8.4f} "
                      f"{ci:>24}  N={int(row['n']):>7,} {pct}")

    # DML
    if "dml" in results:
        dml_r = results["dml"]
        print("\n6. DOUBLE MACHINE LEARNING")
        print("-" * 50)
        print(f"   Treatment effect (female): {dml_r.treatment_effect:.4f}")
        print(f"   Std error:                 {dml_r.std_error:.4f}")
        print(f"   95% CI: [{dml_r.ci_lower:.4f}, {dml_r.ci_upper:.4f}]")
        print(f"   p-value:                   {dml_r.pvalue:.6f}")
        pct = (np.exp(dml_r.treatment_effect) - 1) * 100
        print(f"   Percentage gap:            {pct:.1f}%")

    # Elastic Net
    if "elastic_net" in results:
        en = results["elastic_net"]
        print("\n7. ELASTIC NET — Top interactions")
        print("-" * 50)
        print(f"   Female coef: {en.female_coef:.4f}  |  "
              f"R²: {en.r_squared:.4f}  |  "
              f"Non-zero: {en.n_nonzero}/{en.n_total}")
        top = en.top_interactions.head(10)
        for _, row in top.iterrows():
            print(f"   {row['interaction']:<40} {row['coef']:>10.4f}")

    print("\n" + "=" * 75)


def main():
    # Download ACS PUMS 2022
    df_raw = download_acs_pums(2022)
    logger.info("ACS 2022 raw: %d rows, %d columns", len(df_raw), len(df_raw.columns))

    # Standardize and build features
    df = standardize_and_build_features(df_raw, 2022)

    # Save analysis-ready dataset
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / "acs_2022_analysis_ready.parquet", index=False)
    logger.info("Saved analysis-ready dataset: %d rows", len(df))

    # Run all models
    results = run_all_models(df, RESULTS_DIR)

    # Print results
    print_results(results)

    # Generate plots
    from gender_gap.reporting.charts import generate_all_plots
    plots = generate_all_plots(RESULTS_DIR, RESULTS_DIR / "plots")
    logger.info("Generated %d plots", len(plots))

    # Generate JSON artifact
    from gender_gap.reporting.artifacts import export_json_artifacts
    export_json_artifacts(RESULTS_DIR, RESULTS_DIR)

    logger.info("DONE. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
