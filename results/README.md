# Canonical Results Guide

Use these paths as the canonical outputs from the completed pipeline.

## What Is Canonical

- `results/acs/`
- `results/acs_pooled/`
- `results/cps/`
- `results/sipp/`
- `results/atus/`
- `results/trends/`
- `results/diagnostics/`

## ACS Year-by-Year

- `results/acs/<year>/raw_gap.csv`
- `results/acs/<year>/ols_sequential.csv`
- `results/acs/<year>/oaxaca.csv`
- `results/acs/<year>/quantile_regression.csv`
- `results/acs/<year>/heterogeneity_*.csv`
- `results/acs/<year>/selection_robustness.csv`

## ACS Pooled

- `results/acs_pooled/raw_gap_pooled.csv`
- `results/acs_pooled/ols_pooled.csv`
- `results/acs_pooled/oaxaca_pooled.csv`

## CPS ASEC

- `results/cps/<year>/raw_gap.csv`
- `results/cps/<year>/ols_sequential.csv`
- `results/cps/<year>/selection_robustness.csv`
- `results/trends/cps_asec_raw_gap_trend.csv`
- `results/trends/cps_asec_ols_trend.csv`
- `results/trends/cps_selection_trend.csv`

## ATUS

- `results/atus/time_use_by_gender.csv`

## SIPP

- `results/sipp/2023/raw_gap.csv`
- `results/sipp/2023/ols_sequential.csv`
- `results/sipp/2023/snapshot_summary.csv`
- `results/trends/sipp_monthly_gap_2023.csv`

## Cross-Year Trend Files

- `results/trends/acs_raw_gap_trend.csv`
- `results/trends/acs_raw_gap_trend_with_uncertainty.csv`
- `results/trends/acs_ols_trend.csv`
- `results/trends/acs_oaxaca_trend.csv`
- `results/trends/acs_selection_trend.csv`
- `results/trends/cps_asec_raw_gap_trend.csv`
- `results/trends/cps_asec_ols_trend.csv`

## Trend Charts

- `results/trends/plots/acs_gap_trend.png`
- `results/trends/plots/cross_dataset_gap_trend.png`
- `results/trends/plots/acs_oaxaca_trend.png`

## Diagnostics

- `results/diagnostics/acs_defensibility_diagnostic.csv`
- `results/diagnostics/acs_family_rebuild_comparison.csv`
- `results/diagnostics/acs_oaxaca_sensitivity_yearly.csv`
- `results/diagnostics/acs_oaxaca_sensitivity_summary.csv`
- `results/diagnostics/cps_defensibility_diagnostic.csv`
- `results/diagnostics/artifact_inventory.csv`
- `results/diagnostics/acs_uncertainty_summary.csv`
- `results/diagnostics/context_data_status.csv`
- `results/diagnostics/m6_time_use_bridge_summary.csv`
- `results/diagnostics/robustness_appendix_summary.csv`
- `results/diagnostics/sce_public_series.csv`
- `results/diagnostics/sce_public_summary_metrics.csv`
- `results/diagnostics/sce_subgroup_latest.csv`
- `results/diagnostics/sce_subgroup_spreads.csv`
- `results/diagnostics/sce_measure_map.csv`
- `results/diagnostics/sipp_validation_summary.csv`
- `results/diagnostics/final_synthesis_metrics.csv`
- `results/diagnostics/method_comparison_2023.csv`

## Reports

- `reports/findings_summary.md`
- `reports/defensibility_report.md`
- `reports/acs_family_rebuild_review.md`
- `reports/acs_selection_robustness.md`
- `reports/oaxaca_sensitivity.md`
- `reports/oaxaca_review.md`
- `reports/selection_robustness.md`
- `reports/sce_supplement.md`
- `reports/context_data_status.md`
- `reports/m6_time_use_bridge.md`
- `reports/robustness_appendix.md`
- `reports/sce_public_analysis.md`
- `reports/sce_subgroup_analysis.md`
- `reports/sipp_validation.md`
- `reports/sipp_snapshot.md`
- `reports/sipp_models.md`
- `reports/final_synthesis.md`
- `reports/method_comparison.md`

