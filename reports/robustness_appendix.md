# Robustness Appendix

These checks measure how sampling uncertainty, family controls and employment selection affect the estimated gaps.

## ACS Survey Uncertainty

- ACS raw-gap 90% margins of error average 0.29 percentage points across the full series.
- ACS M5 SDR standard errors average 0.0019.
- Interpretation: the year-by-year ACS headline series is statistically tight; the main uncertainty in this project is methodological rather than sampling noise.

## ACS Family-Field Sensitivity

- Correcting the ACS family variables shifts the year-level M5 coefficient by -0.0032 on average.
- The largest absolute year-level M5 change is 0.0046.
- The pooled ACS P5 coefficient shifts by -0.0042.

## Selection Robustness

- ACS `S2` mean combined expected annual-earnings gap: 38.48%
- ACS `S2` mean IPW worker hourly wage gap: 19.73%
- CPS `S2` mean combined expected annual-earnings gap: 31.99%
- CPS `S2` mean IPW worker hourly wage gap: 19.25%
- Interpretation: annual-earnings gaps are larger than worker-only hourly wage gaps, but reweighting on employment selection does not collapse the worker-gap result.

## Oaxaca Stability

- Mean unexplained share in 2015-2018: 74.08%
- Mean unexplained share in 2019-2023: 87.92%
- Interpretation: decomposition shares become much less stable after 2019 than the sequential OLS estimates, which limits comparisons of decomposition shares across years.

## Context Controls Status

- Cached context sources currently present: 2
- Sources currently requiring manual staging: 1
- Interpretation: BEA RPP and QCEW are available, while OEWS is documented as manual-staging-only from this environment. These sources supply supplemental context variables.

## Source Artifacts

- `results/diagnostics/acs_uncertainty_summary.csv`
- `results/trends/acs_raw_gap_trend_with_uncertainty.csv`
- `results/diagnostics/acs_family_rebuild_comparison.csv`
- `results/trends/acs_selection_trend.csv`
- `results/trends/cps_selection_trend.csv`
- `results/diagnostics/context_data_status.csv`

## Bottom Line

The ACS hourly earnings gap persists after correcting the family controls and reweighting for employment selection.
