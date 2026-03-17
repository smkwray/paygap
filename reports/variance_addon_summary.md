# Variance Addon Summary

- ACS years available in this run: 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024
- Inventory rows tracked: 49
- Missing required inputs: 0

## Reused from repro baseline
- `results/variance/nlsy_validation.csv`

## Newly added for the variance addon
- `diagnostics/variance_local_inventory_report.json`
- `diagnostics/variance_local_inventory_report.md`
- `results/variance/acs_gap_ladder_extended.csv`
- `results/variance/acs_gap_ladder_by_year.csv`
- `results/variance/acs_onet_interactions.csv`
- `results/variance/acs_fertility_risk_penalty.csv`
- `results/variance/acs_fertility_risk_by_quartile.csv`
- `results/variance/acs_same_sex_placebos.csv`
- `results/variance/acs_variance_suite.csv`
- `results/variance/acs_tail_metrics.csv`
- `results/variance/acs_reproductive_dispersion.csv`
- `results/variance/acs_onet_dispersion.csv`
- `results/variance/acs_occupation_dispersion.csv`
- `results/variance/acs_occupation_variability_leaders.csv`
- `results/diagnostics/variance_occupation_harmonization_map.csv`
- `results/diagnostics/variance_onet_merge_coverage.csv`
- `results/variance/acs_selection_corrected_variance.csv`
- `results/variance/atus_mechanisms.csv`
- `reports/atus_variance_mechanisms.md`
- `results/variance/sipp_robustness.csv`
- `results/variance/acs_tail_contrast_summary.csv`
- `results/variance/acs_year_regime_variance_summary.csv`
- `results/variance/acs_soc_group_leaderboard_counts.csv`
- `results/variance/acs_fertility_risk_variance_bridge.csv`

## Missing inputs / skips
- None

## Mechanism / robustness status
- ATUS: ok: reused existing ATUS mechanisms (36 rows)
- SIPP: ok: reused existing SIPP robustness table (10 rows)

## O*NET coverage
- (2013,): 861836/871769 matched (98.9%) via `soc_major_group`
- (2014,): 854420/863691 matched (98.9%) via `soc_major_group`
- (2015,): 856994/866850 matched (98.9%) via `soc_major_group`
- (2016,): 858333/868154 matched (98.9%) via `soc_major_group`
- (2017,): 873024/883158 matched (98.9%) via `soc_major_group`
- (2018,): 884031/887201 matched (99.6%) via `soc_major_group`
- (2019,): 881616/885141 matched (99.6%) via `soc_major_group`
- (2021,): 857287/861048 matched (99.6%) via `soc_major_group`
- (2022,): 900891/905093 matched (99.5%) via `soc_major_group`
- (2023,): 915764/919948 matched (99.5%) via `soc_major_group`
- (2024,): 919033/922891 matched (99.6%) via `soc_major_group`

## Optional validation status
- NLSY79: ready: Processed CFA residual file is available for current NLSY standardizer.
- NLSY97: ready: Processed CFA residual file is available for current NLSY standardizer.
- PSID: ready: A processed main-panel PSID asset is present for paygap.

## Occupation-level variability leaders
- log_annual_earnings_real / female_more_variable_raw #1: occupation `0500` (Agents and business managers of artists, performers, and athletes) (raw_ratio=7.980971828918172, residual_ratio=15.970918276522442, top10_gap_pp=0.6437923080025607, n=2603)
- log_annual_earnings_real / female_more_variable_raw #2: occupation `7260` (Miscellaneous vehicle and mobile equipment mechanics, installers, and repairers) (raw_ratio=4.756679754718989, residual_ratio=4.4052415736279675, top10_gap_pp=0.0277098937022248, n=5011)
- log_annual_earnings_real / female_more_variable_raw #3: occupation `6700` (Elevator and escalator installers and repairers) (raw_ratio=4.068070325934855, residual_ratio=4.066872456711378, top10_gap_pp=-0.0656800244970533, n=2085)
- log_annual_earnings_real / largest_residual_variance_gap #1: occupation `0500` (Agents and business managers of artists, performers, and athletes) (raw_ratio=7.980971828918172, residual_ratio=15.970918276522442, top10_gap_pp=0.6437923080025607, n=2603)
- log_annual_earnings_real / largest_residual_variance_gap #2: occupation `4750` (Parts salespersons) (raw_ratio=3.8151019650427815, residual_ratio=5.153122128679118, top10_gap_pp=-0.7335637015226213, n=7017)
- log_annual_earnings_real / largest_residual_variance_gap #3: occupation `7260` (Miscellaneous vehicle and mobile equipment mechanics, installers, and repairers) (raw_ratio=4.756679754718989, residual_ratio=4.4052415736279675, top10_gap_pp=0.0277098937022248, n=5011)
- log_annual_earnings_real / largest_top10_share_gap #1: occupation `2015` (Probation officers and correctional treatment specialists) (raw_ratio=1.4698376570438405, residual_ratio=1.5648380814982477, top10_gap_pp=-0.7530419348689644, n=7560)
- log_annual_earnings_real / largest_top10_share_gap #2: occupation `4750` (Parts salespersons) (raw_ratio=3.8151019650427815, residual_ratio=5.153122128679118, top10_gap_pp=-0.7335637015226213, n=7017)
- log_annual_earnings_real / largest_top10_share_gap #3: occupation `0500` (Agents and business managers of artists, performers, and athletes) (raw_ratio=7.980971828918172, residual_ratio=15.970918276522442, top10_gap_pp=0.6437923080025607, n=2603)
- log_annual_earnings_real / male_more_variable_raw #1: occupation `2011` (Child, family, and school social workers) (raw_ratio=0.4789675167941292, residual_ratio=0.4256855024743688, top10_gap_pp=-0.0525103585713815, n=3216)
- log_annual_earnings_real / male_more_variable_raw #2: occupation `2005` (Rehabilitation counselors) (raw_ratio=0.4850185582275592, residual_ratio=0.5269513214204212, top10_gap_pp=-0.0135007110589822, n=952)
- log_annual_earnings_real / male_more_variable_raw #3: occupation `split_2010_8950` (Helpers--Production Workers (2010 split bucket)) (raw_ratio=0.4855642468323201, residual_ratio=0.5204723042968223, top10_gap_pp=-0.1712173284407446, n=1271)
- log_hourly_wage_real / female_more_variable_raw #1: occupation `0500` (Agents and business managers of artists, performers, and athletes) (raw_ratio=4.026066833944857, residual_ratio=8.972930708132873, top10_gap_pp=0.6647043500400384, n=2603)
- log_hourly_wage_real / female_more_variable_raw #2: occupation `6250` (Cement masons, concrete finishers, and terrazzo workers) (raw_ratio=3.28669390981258, residual_ratio=3.689947921914304, top10_gap_pp=0.0979442215020361, n=3882)
- log_hourly_wage_real / female_more_variable_raw #3: occupation `6700` (Elevator and escalator installers and repairers) (raw_ratio=2.8752125750240975, residual_ratio=2.8261462252485043, top10_gap_pp=-0.0661283564385065, n=2085)
- log_hourly_wage_real / largest_residual_variance_gap #1: occupation `0500` (Agents and business managers of artists, performers, and athletes) (raw_ratio=4.026066833944857, residual_ratio=8.972930708132873, top10_gap_pp=0.6647043500400384, n=2603)
- log_hourly_wage_real / largest_residual_variance_gap #2: occupation `6250` (Cement masons, concrete finishers, and terrazzo workers) (raw_ratio=3.28669390981258, residual_ratio=3.689947921914304, top10_gap_pp=0.0979442215020361, n=3882)
- log_hourly_wage_real / largest_residual_variance_gap #3: occupation `legacy_2010_8220` (Metal Workers and Plastic Workers, All Other) (raw_ratio=2.665307241823628, residual_ratio=3.4643844731451106, top10_gap_pp=0.0592085048744633, n=13392)
- log_hourly_wage_real / largest_top10_share_gap #1: occupation `2015` (Probation officers and correctional treatment specialists) (raw_ratio=1.1366679063897045, residual_ratio=1.216984213031025, top10_gap_pp=-0.7506721872507168, n=7560)
- log_hourly_wage_real / largest_top10_share_gap #2: occupation `4750` (Parts salespersons) (raw_ratio=1.8873714723613964, residual_ratio=2.5574438861741564, top10_gap_pp=-0.7205939481304636, n=7017)
- log_hourly_wage_real / largest_top10_share_gap #3: occupation `0500` (Agents and business managers of artists, performers, and athletes) (raw_ratio=4.026066833944857, residual_ratio=8.972930708132873, top10_gap_pp=0.6647043500400384, n=2603)
- log_hourly_wage_real / male_more_variable_raw #1: occupation `2910` (Photographers) (raw_ratio=0.3495205807793388, residual_ratio=0.2423209508503552, top10_gap_pp=-0.2882269811242515, n=5419)
- log_hourly_wage_real / male_more_variable_raw #2: occupation `split_2010_8950` (Helpers--Production Workers (2010 split bucket)) (raw_ratio=0.3764071239352898, residual_ratio=0.4378512312210048, top10_gap_pp=-0.1455824165357637, n=1271)
- log_hourly_wage_real / male_more_variable_raw #3: occupation `split_2010_7610` (Helpers--Installation, Maintenance, and Repair Workers (2010 split bucket)) (raw_ratio=0.3888708624083639, residual_ratio=0.4585297566059712, top10_gap_pp=0.0106147541344607, n=577)
