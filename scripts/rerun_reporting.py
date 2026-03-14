#!/usr/bin/env python3
"""Re-run just reporting (plots + artifacts) from existing result CSVs.
Also re-runs Elastic Net and DoubleML with fixes."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    # Load analysis-ready data
    df = pd.read_parquet(DATA_PROCESSED / "acs_2022_analysis_ready.parquet")
    logger.info("Loaded analysis-ready data: %d rows", len(df))

    # ── Re-run Elastic Net ──
    from gender_gap.models.elastic_net import run_elastic_net
    logger.info("Running Elastic Net...")
    try:
        en = run_elastic_net(df, weight_col="person_weight")
        en.top_interactions.to_csv(
            RESULTS_DIR / "elastic_net_interactions.csv", index=False
        )
        logger.info(
            "Elastic Net: female=%.4f, R²=%.4f, %d/%d non-zero",
            en.female_coef, en.r_squared, en.n_nonzero, en.n_total,
        )
        print("\nELASTIC NET RESULTS")
        print("-" * 50)
        print(f"  Female coef: {en.female_coef:.4f}")
        print(f"  R²: {en.r_squared:.4f}")
        print(f"  Non-zero: {en.n_nonzero}/{en.n_total}")
        print(f"  Alpha: {en.alpha:.6f}, L1 ratio: {en.l1_ratio:.2f}")
        print("  Top interactions:")
        for _, row in en.top_interactions.iterrows():
            print(f"    {row['interaction']:<35} {row['coef']:>10.4f}")
    except Exception as e:
        logger.error("Elastic Net failed: %s", e, exc_info=True)

    # ── Re-run DoubleML ──
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
        }]).to_csv(RESULTS_DIR / "dml.csv", index=False)
        pct = (np.exp(dml_result.treatment_effect) - 1) * 100
        print("\nDOUBLEML RESULTS")
        print("-" * 50)
        print(f"  Treatment effect (female): {dml_result.treatment_effect:.4f}")
        print(f"  Std error: {dml_result.std_error:.4f}")
        print(f"  95% CI: [{dml_result.ci_lower:.4f}, {dml_result.ci_upper:.4f}]")
        print(f"  p-value: {dml_result.pvalue:.6f}")
        print(f"  Percentage gap: {pct:.1f}%")
    except Exception as e:
        logger.error("DoubleML failed: %s", e, exc_info=True)

    # ── Generate plots ──
    from gender_gap.reporting.charts import generate_all_plots
    logger.info("Generating plots...")
    plots = generate_all_plots(RESULTS_DIR, RESULTS_DIR / "plots")
    logger.info("Generated %d plots: %s", len(plots), [p.name for p in plots])

    # ── Generate JSON artifacts ──
    from gender_gap.reporting.artifacts import export_json_artifacts
    logger.info("Generating JSON artifacts...")
    export_json_artifacts(RESULTS_DIR, RESULTS_DIR)

    logger.info("DONE. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
