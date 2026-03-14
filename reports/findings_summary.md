# Public-Data Gender Pay Gap: Findings Summary

Generated: 2026-03-13

## Scope

This repo estimates the U.S. male/female earnings gap using free public data:

- ACS PUMS 2015-2019, 2021-2023
- CPS ASEC 2015-2023
- ATUS time-use data

The project reports multiple estimands rather than a single "fully controlled gap":

- total realized gap
- progressively adjusted gap
- employment-selection robustness
- decomposition results
- time-use mechanism evidence

## Headline findings

### ACS year-by-year

- The raw ACS hourly-wage gap is fairly stable, ranging from 15.9% to 18.7%.
- ACS raw-gap survey uncertainty is now available for every year using the 80 ACS replicate weights; 90% intervals are narrow, about plus/minus 0.26 to 0.31 percentage points.
- The ACS M5 adjusted gap is also stable, ranging from about 13.2% to 14.2%.
- After rebuilding ACS from the corrected family fields, the M5 gap becomes slightly larger in every year rather than smaller.
- This means the fully specified ACS model reduces the raw gap, but a sizable residual gap remains in every year.

| Year | Raw gap % | M5 adjusted gap % |
|---|---:|---:|
| 2015 | 18.65 | 14.20 |
| 2016 | 18.43 | 13.65 |
| 2017 | 17.83 | 13.65 |
| 2018 | 17.81 | 13.59 |
| 2019 | 16.95 | 13.86 |
| 2021 | 15.94 | 14.21 |
| 2022 | 17.10 | 13.76 |
| 2023 | 16.81 | 13.23 |

Source files:

- `results/trends/acs_raw_gap_trend.csv`
- `results/trends/acs_raw_gap_trend_with_uncertainty.csv`
- `results/trends/acs_ols_trend.csv`

### ACS pooled panel

- Pooled ACS raw gap: 17.35%
- Pooled ACS P5 adjusted gap: 21.10% in log-point terms, about 19.0% in percent-gap terms if exponentiated from the coefficient
- Pooled ACS Oaxaca unexplained share: 81.54%

The pooled ACS result is larger than the year-by-year M5 gaps because the pooled specification currently uses broad job controls in order to keep the design matrix tractable over 7.0M rows. That pooled result is useful, but the year-by-year ACS M5 trend is the more conservative headline series.

Source files:

- `results/acs_pooled/raw_gap_pooled.csv`
- `results/acs_pooled/ols_pooled.csv`
- `results/acs_pooled/oaxaca_pooled.csv`

### CPS ASEC cross-check

- CPS ASEC raw gaps range from 15.8% to 20.6%.
- CPS ASEC fully specified gaps remain larger than ACS, about 17.0% to 20.8%.

That is directionally consistent with ACS: a substantial residual gap remains after observable controls. The CPS levels are somewhat larger, which is plausible given different wage measurement, sample construction, and the lighter CPS control surface in this pipeline.

Source files:

- `results/trends/cps_asec_raw_gap_trend.csv`
- `results/trends/cps_asec_ols_trend.csv`

### CPS selection robustness

The new CPS selection-robustness layer reinforces the main interpretation that worker wage gaps and total earnings gaps are different estimands.

- In the richest CPS selection block (`S2`), the mean `2015-2023` combined expected annual-earnings gap is 31.99%.
- The mean observed total annual-earnings gap is 27.43%.
- The mean observed worker hourly wage gap is 19.51%.
- The mean IPW-reweighted worker hourly wage gap is 19.25%.

That pattern is substantively useful: labor-force selection matters for total annual earnings, but it does not collapse the worker-only hourly wage gap.

Source files:

- `results/trends/cps_selection_trend.csv`
- `reports/selection_robustness.md`

### ACS selection robustness

The ACS selection-robustness layer now tells a similar but sharper story on annual earnings versus worker wages.

- In the richest ACS selection block (`S2`), the mean `2015-2023` combined expected annual-earnings gap is 38.48%.
- The mean observed total annual-earnings gap is 36.55%.
- The mean observed worker hourly wage gap is 18.83%.
- The mean IPW-reweighted worker hourly wage gap is 19.73%.

That is substantively useful in two ways: ACS and CPS agree that employment selection matters more for annual earnings than for worker-only hourly wages, and the worker-gap result still remains large after this sensitivity check.

Source files:

- `results/trends/acs_selection_trend.csv`
- `reports/acs_selection_robustness.md`

### ATUS mechanism evidence

ATUS results support a work/family time-allocation channel:

- women spend about 68 fewer minutes/day in paid work
- women spend about 32 more minutes/day in housework
- women spend about 11 more minutes/day in childcare
- women spend about 8 fewer minutes/day in commute-related travel

This is mechanism evidence, not a person-level control merged into ACS/CPS.

The repo now also has a formal mechanism-bridge note in `reports/m6_time_use_bridge.md` with compact metrics in `results/diagnostics/m6_time_use_bridge_summary.csv`. That is the defensible `M6` surface here: an ATUS-based interpretation layer that follows the ACS/CPS residual-gap and selection results, not a literal merged regression.

Source file:

- `results/atus/time_use_by_gender.csv`

### SCE supplemental expectations evidence

The repo now includes a methodological SCE supplement, a first empirical public-series note, and a subgroup extension from the official chart workbook.

- SCE is the right public source here for reservation wages, expected offer wages, offer receipt, and offer acceptance outcomes.
- SCE should be used as supporting evidence on expectations and outside options, not as a literal merged control inside ACS/CPS wage regressions.
- The conceptual supplement is in `reports/sce_supplement.md`, with a structured measure inventory in `results/diagnostics/sce_measure_map.csv`.
- The empirical public-series note is in `reports/sce_public_analysis.md`, and the subgroup extension is in `reports/sce_subgroup_analysis.md`.
- In the latest official public SCE chart data wave (`2025-11`), men report both a higher expected offer wage (`73.37` vs `55.91`) and a higher reservation wage (`91.12` vs `69.78`) than women.
- Those gender gaps are persistent across all public waves in the chart workbook, but the education and income spreads are even larger than the gender spread in the latest wave.

## Important interpretive points

### What looks robust

- The raw gap is persistent across ACS and CPS.
- The adjusted ACS gap remains around 13% to 14% across all available years.
- The adjusted CPS gap remains materially above zero in every year.
- ATUS is directionally consistent with time-allocation and care-burden channels.

### What needs caution

- Oaxaca unexplained shares rise sharply after 2019:
  - 2015-2018: about 73% to 75%
  - 2019: 81.77%
  - 2021: 93.62%
  - 2022: 88.48%
  - 2023: 87.81%

This may reflect a real post-2019 change in the explanatory power of the simple Oaxaca control set, but it could also reflect decomposition fragility. It should not be treated as the cleanest headline result until reviewed more carefully.

The newer Oaxaca sensitivity note now narrows that concern: the post-2019 unexplained-share jump is driven more by the explained component collapsing under a fairly stable total gap than by the total wage gap itself exploding. That still makes Oaxaca secondary to the ACS M5 trend, but it is a more specific caution than “pipeline may be broken.”

- The pooled ACS model uses broad occupation and industry controls, while the year-by-year ACS models use detailed codes. That makes pooled and year-specific adjusted gaps not perfectly comparable.

- Top-level `results/*.csv` files appear to be legacy single-run artifacts from earlier work. The canonical finished outputs are the year directories plus `results/acs_pooled`, `results/cps`, `results/atus`, and `results/trends`.

## Defensibility assessment

### What is already strong

- Public-data-only design
- Multiple datasets, not a single-file claim
- Multiple model families
- Year-by-year and pooled ACS views
- External cross-check with CPS
- CPS employment-selection robustness layer
- ACS employment-selection robustness layer
- Separate mechanism evidence via ATUS
- Implemented SCE supplement on reservation wages and job-offer expectations
- Real public-use SIPP file now stages and standardizes successfully, with a validation artifact documenting coverage
- SIPP now also has a first descriptive analysis artifact for the 2023 public-use file
- SIPP now also has a modest adjusted-gap surface for the 2023 public-use file
- ACS replicate-weight acquisition completed for 2015-2019 and 2021-2023
- ACS SDR uncertainty is now complete for the full ACS analysis series, not just a single-year snapshot
- Consolidated robustness appendix in `reports/robustness_appendix.md`
- Reproducible pipeline and tests

### What still limits publication-quality defensibility

- The current robustness appendix is consolidated and useful, but it is still a summary layer rather than a full paper-style appendix with additional alternative samples and control sets
- No full within-SCE microdata model yet; the current SCE layer now includes a public-series empirical note plus staged public files, but not a richer microdata regression workflow
- No SIPP deep-control file yet
- The current SIPP layer is validated and useful for extension work, but it is not yet integrated into the main wage-model stack
- The current SIPP modeling layer is still modest; it is not yet a covariate-rich or longitudinal SIPP module
- The ACS family-field rebuild is complete, but its implications now need to be reflected throughout the final narrative and robustness discussion

### ACS uncertainty snapshot

- ACS raw-gap and M5 SDR uncertainty are now complete across 2015-2019 and 2021-2023.
- ACS raw-gap 90% margins of error are narrow in every year, about 0.26 to 0.31 percentage points, with a mean of 0.29.
- ACS M5 SDR standard errors range from about 0.0017 to 0.0022, with a mean of 0.0019.
- ACS 2023 raw gap: 16.81%
  SDR SE: 0.19 percentage points
  90% CI: 16.50% to 17.12%
- ACS 2023 M5 female coefficient: -0.1408
  SDR SE: 0.0017
  90% CI: -0.1436 to -0.1379

## Recommended next work

1. Thread the completed all-year ACS SDR intervals into the public-facing tables and defensibility artifacts.
2. If you want to keep pushing decomposition work, benchmark Oaxaca against broader aggregation or alternative sample restrictions before making it prominent.
3. If desired, extend SCE beyond the public-series note into a richer within-SCE microdata workflow.
4. Add any remaining appendix extensions you want beyond the current consolidated robustness appendix:
   - full-time/full-year sample
   - alternative winsorization
   - alternative occupation/industry aggregation
   - ACS vs CPS comparison table
5. Add a short note or appendix table documenting that the corrected ACS family fields modestly increase the year-by-year M5 gap rather than reducing it.

## Bottom line

The core project is working and already supports a defensible main conclusion:

> Using large public U.S. datasets, the observed hourly earnings gap between men and women is substantial, and a meaningful residual gap remains after standard observable controls. Time-use evidence is consistent with family and schedule-related mechanisms, but those mechanisms do not eliminate the residual gap.

The remaining work is less about basic buildout and more about hardening the project for scrutiny.
