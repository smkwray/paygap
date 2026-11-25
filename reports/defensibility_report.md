# Wage estimates and survey diagnostics

## Headline Results

- ACS latest-year raw gap (2023): 16.81%
- ACS latest-year M5 adjusted gap (2023): -0.1419 in log points, about 13.23% as an exponentiated percent gap
- ACS pooled raw gap: 17.35%
- ACS pooled P5 adjusted gap: -0.2110 in log points, about 19.03% as an exponentiated percent gap
- CPS latest-year raw gap (2023): 15.77%
- CPS latest-year M_full adjusted gap (2023): -16.96%

## Main Cautions

- The largest ACS Oaxaca unexplained share occurs in 2021: 93.62%.
- Oaxaca unexplained shares rise sharply after 2019, while the M5 adjusted gaps stay comparatively stable. The new sensitivity note indicates that this is mostly an explained-component collapse under a fairly stable total gap, which still makes the decomposition more fragile than the sequential OLS trend.
- The pooled ACS model uses broad occupation and industry controls for tractability, so pooled and year-specific adjusted gaps should not be treated as identical estimands.
- Correcting the ACS family fields modestly increased the year-by-year M5 gap in every year rather than attenuating it.
- The SCE layer now includes an empirical public-series note, but there is still no richer within-SCE microdata regression workflow.

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
