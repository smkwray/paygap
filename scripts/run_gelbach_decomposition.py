#!/usr/bin/env python3
"""Run Gelbach (2016) decomposition on ACS data.

Produces an order-invariant attribution of how much each covariate block
moves the female coefficient.  Runs on individual ACS years (not pooled)
to stay within the dense-matrix row limit.

Usage:
    python scripts/run_gelbach_decomposition.py              # all available years
    python scripts/run_gelbach_decomposition.py --year 2023  # single year
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gender_gap.models.gelbach import (
    DEFAULT_GELBACH_BLOCKS,
    REPRODUCTIVE_GELBACH_BLOCKS,
    gelbach_decomposition,
    gelbach_to_dataframe,
)
from gender_gap.repro import _load_acs_year, _resolve_onet_dir
from gender_gap.features.occupation_context import build_onet_indices, merge_onet_context
from gender_gap.features.sample_filters import filter_prime_age_wage_salary
from gender_gap.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "gelbach"
ACS_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]


def run_gelbach_year(
    year: int,
    onet_indices: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run default and reproductive Gelbach decompositions for one ACS year."""
    logger.info("Loading ACS %d ...", year)
    df = _load_acs_year(year)
    if df is None:
        logger.warning("ACS %d: no data found, skipping", year)
        return {}

    df = filter_prime_age_wage_salary(df)
    logger.info("ACS %d: %d prime-age workers", year, len(df))

    if len(df) < 200:
        logger.warning("ACS %d: too few rows (%d), skipping", year, len(df))
        return {}

    # Compute log wage if not present
    if "log_hourly_wage_real" not in df.columns:
        import numpy as np
        hw = pd.to_numeric(df["hourly_wage_real"], errors="coerce").clip(lower=0.01)
        df["log_hourly_wage_real"] = np.log(hw)

    tables = {}

    # --- Default decomposition (M1 → M5) ---
    try:
        result_default = gelbach_decomposition(
            df,
            outcome="log_hourly_wage_real",
            blocks=DEFAULT_GELBACH_BLOCKS,
        )
        tables["default"] = gelbach_to_dataframe(result_default)
        tables["default"]["year"] = year
        tables["default"]["specification"] = "default_M1_to_M5"
        logger.info(
            "ACS %d default: base=%.4f, full=%.4f, identity=%.2e",
            year, result_default.base_coef, result_default.full_coef,
            result_default.identity_check,
        )
    except Exception as e:
        logger.error("ACS %d default decomposition failed: %s", year, e)

    # --- Reproductive decomposition (M1 → M7) ---
    if onet_indices is not None:
        try:
            df_onet, _ = merge_onet_context(df, onet_indices)
            result_repro = gelbach_decomposition(
                df_onet,
                outcome="log_hourly_wage_real",
                blocks=REPRODUCTIVE_GELBACH_BLOCKS,
            )
            tables["reproductive"] = gelbach_to_dataframe(result_repro)
            tables["reproductive"]["year"] = year
            tables["reproductive"]["specification"] = "reproductive_M1_to_M7"
            logger.info(
                "ACS %d reproductive: base=%.4f, full=%.4f, identity=%.2e",
                year, result_repro.base_coef, result_repro.full_coef,
                result_repro.identity_check,
            )
        except Exception as e:
            logger.error("ACS %d reproductive decomposition failed: %s", year, e)

    return tables


def main():
    parser = argparse.ArgumentParser(description="Gelbach decomposition on ACS data")
    parser.add_argument("--year", type=int, default=None, help="Single ACS year (default: all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load O*NET indices if available
    onet_indices = None
    recipe_path = PROJECT_ROOT / "configs" / "onet_index_recipe.yaml"
    try:
        onet_dir = _resolve_onet_dir(["Work Context.txt"])
        if (onet_dir / "Work Context.txt").exists() and recipe_path.exists():
            onet_indices = build_onet_indices(onet_dir, recipe_path)
            logger.info("O*NET indices loaded: %d SOC groups", len(onet_indices))
    except Exception as e:
        logger.warning("O*NET loading failed, skipping reproductive blocks: %s", e)

    years = [args.year] if args.year else ACS_YEARS
    all_default = []
    all_repro = []

    for year in years:
        tables = run_gelbach_year(year, onet_indices=onet_indices)
        if "default" in tables:
            all_default.append(tables["default"])
        if "reproductive" in tables:
            all_repro.append(tables["reproductive"])

    # Write results
    if all_default:
        default_df = pd.concat(all_default, ignore_index=True)
        default_df.to_csv(RESULTS_DIR / "gelbach_default_by_year.csv", index=False)
        logger.info("Wrote %s", RESULTS_DIR / "gelbach_default_by_year.csv")

        # Write markdown summary
        _write_markdown_summary(default_df, "default", RESULTS_DIR / "gelbach_default_summary.md")

    if all_repro:
        repro_df = pd.concat(all_repro, ignore_index=True)
        repro_df.to_csv(RESULTS_DIR / "gelbach_reproductive_by_year.csv", index=False)
        logger.info("Wrote %s", RESULTS_DIR / "gelbach_reproductive_by_year.csv")

        _write_markdown_summary(repro_df, "reproductive", RESULTS_DIR / "gelbach_reproductive_summary.md")

    if not all_default and not all_repro:
        logger.error("No decompositions succeeded. Check data availability.")
        return 1

    return 0


def _write_markdown_summary(df: pd.DataFrame, spec_label: str, path: Path):
    """Write a human-readable markdown summary of Gelbach results."""
    lines = [
        f"# Gelbach Decomposition — {spec_label}",
        "",
        "Order-invariant attribution of how much each covariate block moves the",
        "female coefficient (Gelbach 2016). Negative delta = block absorbs part of",
        "the female penalty (makes the coefficient less negative).",
        "",
    ]

    years = sorted(df["year"].unique())
    block_rows = df[df["block"] != "TOTAL"]
    blocks = [b for b in block_rows["block"].unique() if b != "TOTAL"]

    # Per-year tables
    for year in years:
        yr_data = df[df["year"] == year]
        total_row = yr_data[yr_data["block"] == "TOTAL"]
        total_explained = total_row["delta"].iloc[0] if len(total_row) else float("nan")

        lines.append(f"## {year}")
        lines.append("")
        lines.append(f"Total explained (base - full): {total_explained:.4f}")
        lines.append("")
        lines.append("| Block | Delta | SE | % of Explained |")
        lines.append("|-------|------:|---:|---------------:|")

        for _, row in yr_data[yr_data["block"] != "TOTAL"].iterrows():
            pct = row["pct_of_explained"]
            pct_str = f"{pct:.1f}%" if pd.notna(pct) else "—"
            se_str = f"{row['se']:.4f}" if pd.notna(row["se"]) else "—"
            lines.append(f"| {row['block']} | {row['delta']:.4f} | {se_str} | {pct_str} |")

        lines.append("")

    # Cross-year stability
    if len(years) > 1:
        lines.append("## Cross-Year Stability")
        lines.append("")
        lines.append("| Block | Mean Delta | SD Delta | Min | Max |")
        lines.append("|-------|----------:|---------:|----:|----:|")
        for block in blocks:
            bdata = block_rows[block_rows["block"] == block]["delta"]
            lines.append(
                f"| {block} | {bdata.mean():.4f} | {bdata.std():.4f} | "
                f"{bdata.min():.4f} | {bdata.max():.4f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    sys.exit(main())
