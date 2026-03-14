# Defensibility Report

## Status

- Core pipeline status: complete
- Purpose of this report: summarize what is currently defensible, what is provisional, and what still needs hardening.

## Headline Results

- ACS latest-year raw gap (2023): 16.81%
- ACS latest-year M5 adjusted gap (2023): -0.1419 in log points, about 13.23% as an exponentiated percent gap
- ACS pooled raw gap: 17.35%
- ACS pooled P5 adjusted gap: -0.2110 in log points, about 19.03% as an exponentiated percent gap
- CPS latest-year raw gap (2023): 15.77%
- CPS latest-year M_full adjusted gap (2023): -16.96%

## What Is Already Defensible

- The project now runs end-to-end on multiple public datasets rather than relying on a single file.
- ACS year-by-year results are stable: the adjusted gap remains in a narrow band around 13% to 14%.
- CPS ASEC shows the same qualitative pattern: a substantial residual gap remains after controls.
- CPS now also has a dedicated employment-selection robustness layer, which separates worker wage-gap claims from total annual-earnings claims.
- ACS now also has a completed employment-selection robustness layer across all 8 analysis years.
- ACS year-level and pooled outputs have now been rebuilt from the corrected family fields rather than the old all-zero placeholders.
- ATUS provides independent mechanism evidence on paid work, housework, childcare, and commute-related travel.
- The repo now includes an SCE supplement on reservation wages, offer expectations, and job-offer dynamics as a separate mechanism-evidence module.
- The repo now also has a validated real-data SIPP standardization lane rather than only a planned SIPP extension.
- The pooled ACS step now completes on 7.0M rows using broad job controls.

## Main Cautions

- The largest ACS Oaxaca unexplained share occurs in 2021: 93.62%.
- Oaxaca unexplained shares rise sharply after 2019, while the M5 adjusted gaps stay comparatively stable. The new sensitivity note indicates that this is mostly an explained-component collapse under a fairly stable total gap, which still makes the decomposition more fragile than the sequential OLS trend.
- The pooled ACS model uses broad occupation and industry controls for tractability, so pooled and year-specific adjusted gaps should not be treated as identical estimands.
- Correcting the ACS family fields modestly increased the year-by-year M5 gap in every year rather than attenuating it.
- The SCE layer now includes an empirical public-series note, but there is still no richer within-SCE microdata regression workflow.
- The SIPP layer is validated on a real public-use file, but it is not yet part of the main wage-model stack.

## ACS Replicate-Weight Feasibility

The original ACS API parquet files still contain only the main person weight (`PWGTP`), but the replicate-weight acquisition lane is now complete for the ACS analysis years.
That means survey-consistent ACS uncertainty is now feasible from the current raw extracts.

- `acs_pums_2015_api.parquet`: 0 ACS replicate-weight columns present
- `acs_pums_2016_api.parquet`: 0 ACS replicate-weight columns present
- `acs_pums_2017_api.parquet`: 0 ACS replicate-weight columns present
- `acs_pums_2018_api.parquet`: 0 ACS replicate-weight columns present
- `acs_pums_2019_api.parquet`: 0 ACS replicate-weight columns present
- `acs_pums_2021_api.parquet`: 0 ACS replicate-weight columns present
- `acs_pums_2022_api.parquet`: 0 ACS replicate-weight columns present
- `acs_pums_2023_api.parquet`: 0 ACS replicate-weight columns present
- `acs_pums_2015_api_repweights.parquet`: 80 ACS replicate-weight columns present
- `acs_pums_2016_api_repweights.parquet`: 80 ACS replicate-weight columns present
- `acs_pums_2017_api_repweights.parquet`: 80 ACS replicate-weight columns present
- `acs_pums_2018_api_repweights.parquet`: 80 ACS replicate-weight columns present
- `acs_pums_2019_api_repweights.parquet`: 80 ACS replicate-weight columns present
- `acs_pums_2021_api_repweights.parquet`: 80 ACS replicate-weight columns present
- `acs_pums_2022_api_repweights.parquet`: 80 ACS replicate-weight columns present
- `acs_pums_2023_api_repweights.parquet`: 80 ACS replicate-weight columns present

## ACS Uncertainty Snapshot

- ACS raw gap (2023) with SDR uncertainty: 16.81% (SE 0.19, 90% CI 16.50% to 17.12%)
- ACS M5 female coefficient (2023) with SDR uncertainty: -0.1408 (SE 0.0017, 90% CI -0.1436 to -0.1379)
- ACS raw-gap uncertainty trend is complete for 8 years; average 90% margin of error is 0.29 percentage points

## ATUS Mechanism Snapshot

- `minutes_paid_work_diary`: female-male difference = -67.96 minutes/day
- `minutes_housework`: female-male difference = 31.91 minutes/day
- `minutes_childcare`: female-male difference = 11.37 minutes/day
- `minutes_commute_related_travel`: female-male difference = -8.07 minutes/day
- `reports/m6_time_use_bridge.md` now formalizes the defensible `M6` layer as a mechanism-aware bridge, not a merged ACS+ATUS wage regression.

## CPS Selection Snapshot

- CPS `S2` mean combined expected annual-earnings gap (2015-2023): 31.99%
- CPS `S2` mean observed total annual-earnings gap (2015-2023): 27.43%
- CPS `S2` mean observed worker hourly wage gap (2015-2023): 19.51%
- CPS `S2` mean IPW worker hourly wage gap (2015-2023): 19.25%

## ACS Selection Snapshot

- ACS `S2` mean female employment probability effect (2015-2023): -8.76 percentage points
- ACS `S2` mean combined expected annual-earnings gap (2015-2023): 38.48%
- ACS `S2` mean observed total annual-earnings gap (2015-2023): 36.55%
- ACS `S2` mean observed worker hourly wage gap (2015-2023): 18.83%
- ACS `S2` mean IPW worker hourly wage gap (2015-2023): 19.73%

## SCE Supplemental Status

- `reports/sce_supplement.md` documents the defensible role of the New York Fed SCE Labor Market Survey in this repo.
- `results/diagnostics/sce_measure_map.csv` maps reservation wage, offer expectations, offer receipt, and search-intensity constructs to their best use and limits.
- `reports/sce_public_analysis.md` and `results/diagnostics/sce_public_summary_metrics.csv` now provide a first empirical public-series SCE layer.
- `reports/sce_subgroup_analysis.md`, `results/diagnostics/sce_subgroup_latest.csv`, and `results/diagnostics/sce_subgroup_spreads.csv` extend that layer to subgroup differences by sex, age, education, and income from the official chart workbook.
- The current SCE layer is intentionally supplemental: it supports bargaining and expectations interpretation but is not merged into the ACS/CPS regression surface.

## SIPP Status

- `data/processed/sipp_standardized.parquet` now contains the first real standardized Census public-use SIPP file in the repo.
- `reports/sipp_validation.md` and `results/diagnostics/sipp_validation_summary.csv` document coverage for wages, hours, occupation, and industry on that file.
- `reports/sipp_snapshot.md`, `results/sipp/2023/snapshot_summary.csv`, and `results/trends/sipp_monthly_gap_2023.csv` now provide a first descriptive analysis layer for the 2023 public-use file.
- `reports/sipp_models.md` and `results/sipp/2023/ols_sequential.csv` now add a modest SIPP-specific adjusted-gap surface: the 2023 raw hourly gap is 15.09% and the `SIPP3` adjusted gap is about 10.91%.
- The current SIPP lane is ready for extension work, but it still stops well short of a full longitudinal or covariate-rich SIPP analysis stack.

## Robustness Appendix Status

- `reports/robustness_appendix.md` now consolidates the completed uncertainty, family-rebuild, selection, Oaxaca-stability, and context-status surfaces.
- `results/diagnostics/robustness_appendix_summary.csv` provides a compact machine-readable summary of the same checks.
- `reports/oaxaca_sensitivity.md` and `results/diagnostics/acs_oaxaca_sensitivity_summary.csv` now quantify the post-2019 decomposition shift directly.

## Recommended Next Work

1. Reconcile the rebuilt ACS headline coefficients with the ACS SDR uncertainty layer so the narrative is using one fully current ACS surface.
2. If you want deeper decomposition work, benchmark Oaxaca against broader aggregation or alternative sample restrictions before making decomposition claims prominent in any external memo.
3. If you want deeper empirical bargaining estimates, extend the staged SCE public files into a richer within-SCE microdata workflow.
4. If you want to extend SIPP, build a first SIPP-specific modeling/reporting layer off the now-validated standardized parquet.
5. Extend the appendix only if you want additional sample or control sensitivity checks beyond the current consolidated appendix.
6. Add a compact comparison table documenting how the corrected ACS family fields changed year-by-year M5 and pooled P5 estimates.
