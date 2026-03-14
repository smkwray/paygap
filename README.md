# U.S. Gender Earnings Gap — Public Data Analysis

This repo estimates the male/female hourly earnings gap in the United States using
publicly available federal survey data. It runs multiple estimation methods across
multiple datasets and years, reports both unadjusted and adjusted gaps, and
includes mechanism evidence from time-use and labor-market expectations surveys.

## Cross-dataset headline (2023)

| Dataset | Raw hourly gap | Adjusted hourly gap | Model | N |
|---------|---------------:|--------------------:|-------|--:|
| ACS     | 16.81%         | 13.23%              | M5    | ~900K/yr |
| CPS ASEC| 15.77%         | 16.96%              | M_full| ~60K/yr |
| SIPP    | 15.09%         | 10.91%              | SIPP3 | 182,658 |

- ACS 2023 raw gap 90% CI: 16.50% – 17.12% (successive-difference replication, 80 replicate weights)
- ACS adjusted gap is stable at 13–14% across all available years (2015–2019, 2021–2023)
- All three datasets show a residual gap after observable controls

## Results

<details>
<summary><strong>ACS year-by-year trend</strong></summary>

The ACS is the primary headline dataset. Sequential OLS progressively adds control
blocks: demographics (M1), geography (M2), job sorting (M3), schedule/commute (M4),
and family (M5).

| Year | Raw gap % | M5 adjusted gap % |
|-----:|----------:|-------------------:|
| 2015 | 18.65     | 14.20              |
| 2016 | 18.43     | 13.65              |
| 2017 | 17.83     | 13.65              |
| 2018 | 17.81     | 13.59              |
| 2019 | 16.95     | 13.86              |
| 2021 | 15.94     | 14.21              |
| 2022 | 17.10     | 13.76              |
| 2023 | 16.81     | 13.23              |

- Raw gap ranges from 15.9% to 18.7% across years
- M5 adjusted gap stays in a narrow 13.2–14.2% band
- ACS raw-gap 90% margins of error average 0.29 percentage points (tight)
- ACS M5 SDR standard errors average 0.0019

</details>

<details>
<summary><strong>ACS pooled panel (7M+ observations)</strong></summary>

- Pooled raw gap: 17.35%
- Pooled P5 adjusted gap: ~19.0% (−0.2110 log points)
- Pooled Oaxaca unexplained share: 81.54%

The pooled result is larger than year-by-year M5 because the pooled specification
uses broad occupation/industry controls for tractability over 7.0M rows. The
year-by-year M5 trend is the more conservative headline series.

</details>

<details>
<summary><strong>CPS ASEC cross-check</strong></summary>

- CPS raw gaps range from 15.8% to 20.6% across 2015–2023
- CPS fully-specified adjusted gaps range from 17.0% to 20.8%
- Directionally consistent with ACS: a substantial residual gap remains after controls
- CPS levels run somewhat larger, which is expected given different wage measurement,
  sample construction, and a lighter control surface

</details>

<details>
<summary><strong>SIPP cross-check (2023)</strong></summary>

SIPP specification ladder with available variables:

| Model | Controls added | Female coef | Gap % | R² |
|-------|----------------|------------:|------:|---:|
| SIPP0 | female only | −0.1448 | 13.48 | 0.006 |
| SIPP1 | + month | −0.1448 | 13.48 | 0.007 |
| SIPP2 | + occupation, industry | −0.1050 | 9.97 | 0.245 |
| SIPP3 | + hours, paid-hourly, multi-job | −0.1156 | 10.91 | 0.253 |

Raw hourly gap: 15.09%. Adjusted gap after available controls: 10.91%.

</details>

<details>
<summary><strong>Method comparison (OLS vs DML vs Oaxaca, 2023)</strong></summary>

These are not identical estimands. OLS reports a conditional female coefficient,
DML reports a residual effect after flexible nuisance-model adjustment, and Oaxaca
decomposes the gap into explained/unexplained components.

| Dataset  | Raw gap % | OLS adjusted % | DML adjusted % | Oaxaca unexplained % |
|----------|----------:|---------------:|---------------:|---------------------:|
| ACS      | 16.81     | 13.23          | 17.88          | 87.81                |
| CPS ASEC | 15.77     | 16.96          | 19.61          | 90.88                |
| SIPP     | 15.09     | 10.91          | 12.89          | 85.60                |

DML runs larger than OLS in all three datasets — flexible residualization does not
drive the female effect toward zero. Oaxaca answers a different question and is
treated as supplemental.

</details>

<details>
<summary><strong>Oaxaca-Blinder decomposition trend</strong></summary>

| Year | Total log gap | Explained % | Unexplained % | M5 adjusted gap % |
|-----:|:-------------:|:-----------:|:-------------:|-------------------:|
| 2015 | 0.1794        | 26.47       | 73.53         | 14.20              |
| 2016 | 0.1731        | 25.48       | 74.52         | 13.65              |
| 2017 | 0.1708        | 26.13       | 73.87         | 13.65              |
| 2018 | 0.1702        | 25.62       | 74.38         | 13.59              |
| 2019 | 0.1692        | 18.23       | 81.77         | 13.86              |
| 2021 | 0.1639        | 6.38        | 93.62         | 14.21              |
| 2022 | 0.1654        | 11.52       | 88.48         | 13.76              |
| 2023 | 0.1623        | 12.19       | 87.81         | 13.23              |

The unexplained share rises sharply after 2019 (from ~74% to ~88–94%), while the
M5 adjusted gap stays stable. The post-2019 shift is driven by the explained component
collapsing under a stable total gap, not by the total gap exploding. This makes
Oaxaca secondary to the sequential OLS trend for headline reporting.

</details>

<details>
<summary><strong>Employment-selection robustness</strong></summary>

IPW separates worker-only wage gaps from total earnings gaps (including non-workers).

**ACS S2 block (mean 2015–2023):**
| Measure | Gap |
|---------|----:|
| Combined expected annual earnings | 38.48% |
| Observed total annual earnings | 36.55% |
| Observed worker hourly wage | 18.83% |
| IPW-reweighted worker hourly wage | 19.73% |

**CPS S2 block (mean 2015–2023):**
| Measure | Gap |
|---------|----:|
| Combined expected annual earnings | 31.99% |
| Observed total annual earnings | 27.43% |
| Observed worker hourly wage | 19.51% |
| IPW-reweighted worker hourly wage | 19.25% |

Employment selection matters for annual earnings but does not collapse the
worker-only hourly wage gap. ACS and CPS agree on this pattern.

</details>

<details>
<summary><strong>ATUS time-use mechanism evidence</strong></summary>

ATUS provides separate mechanism evidence on daily time allocation. It is not merged
into ACS/CPS wage regressions; it is used to interpret the residual gap.

| Activity | Female − Male (min/day) |
|----------|------------------------:|
| Paid work | −67.96 |
| Housework | +31.91 |
| Childcare | +11.37 |
| Commute-related travel | −8.07 |
| Total unpaid (housework + childcare) | +43.28 |
| Net paid + unpaid | −24.68 |

Women spend ~68 fewer minutes/day in paid work and ~43 more minutes/day in unpaid
household and childcare work. This is consistent with schedule and care-burden
channels but does not eliminate the residual worker wage gap.

</details>

<details>
<summary><strong>SCE expectations and reservation wages</strong></summary>

The NY Fed Survey of Consumer Expectations provides data on expected offer wages and
reservation wages by subgroup.

**Latest wave (2025-11):**

| Measure | Women | Men | Gap |
|---------|------:|----:|----:|
| Expected offer wage | $55.91 | $73.37 | $17.46 |
| Reservation wage | $69.78 | $91.12 | $21.34 |

Men are above women in every public wave for both measures (100% persistence).

**Subgroup comparison (latest wave):**

| Subgroup spread | Expected offer gap | Reservation wage gap |
|-----------------|-------------------:|---------------------:|
| Gender (M − F)  | $17.46             | $21.34               |
| Education (college+ − less) | $40.23 | $36.94               |
| Income (>$60K − ≤$60K) | $42.38     | $52.57               |

Education and income gradients are larger than the gender gradient. Gender
differences in expected and reservation wages exist inside even larger gradients
by education and income.

</details>

<details>
<summary><strong>NLSY cohort sub-analysis</strong></summary>

NLSY adds cognitive ability (g-factor from ASVAB) and richer life-course controls
that ACS/CPS cannot provide.

**NLSY79 (year-2000 earnings, N=2,890):**
- Raw annual-earnings gap: 44.43%
- Occupation sorting reduction: 7.02 pp
- Skills/traits reduction: 5.62 pp
- Family-background reduction: 0.04 pp
- Final deep-model gap: 34.10%

**NLSY97 (year-2019 earnings, N=2,486):**
- Raw annual-earnings gap: 34.19%
- Adult-resources reduction: 6.60 pp
- Skills/achievement reduction: 1.65 pp
- Family-background reduction: −0.06 pp
- Final deep-model gap: 31.89%

Skills, background, occupation, and family/resource variables each reduce the gap
somewhat, but meaningful residual gaps remain after all blocks are added. The
adult-resources block (household income, net worth) is post-market and should be
read as mechanism-sensitive accounting, not a pre-market explanation.

</details>

<details>
<summary><strong>Robustness summary</strong></summary>

- **Survey uncertainty:** ACS raw-gap 90% margins of error average 0.29 pp. M5
  standard errors average 0.0019. Sampling noise is not the main source of
  uncertainty.
- **Family-field sensitivity:** Correcting ACS family variables shifts the M5
  coefficient by −0.0032 on average (largest year-level shift: 0.0046). The bug
  was real and worth fixing but does not overturn results.
- **Selection robustness:** Annual-earnings gaps are larger than worker-only hourly
  gaps, but IPW reweighting does not collapse the worker gap.
- **Oaxaca stability:** Decomposition shares become unstable after 2019 while OLS
  estimates remain stable. Oaxaca is secondary to sequential OLS for reporting.
- **Cross-dataset agreement:** ACS, CPS, and SIPP all show residual gaps after
  their respective control surfaces.

</details>

## Data sources

All data are free, public, and U.S. federal:

| Dataset | Agency | Years | Role |
|---------|--------|-------|------|
| [ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html) | Census Bureau | 2015–2019, 2021–2023 | Primary wage estimates |
| [CPS ASEC](https://www.census.gov/programs-surveys/cps.html) | Census/BLS | 2015–2023 | Cross-check + selection robustness |
| [SIPP](https://www.census.gov/programs-surveys/sipp.html) | Census Bureau | 2023 | Additional cross-check |
| [ATUS](https://www.bls.gov/tus/) | BLS | Pooled | Time-use mechanism evidence |
| [SCE Labor Market Survey](https://www.newyorkfed.org/microeconomics/sce/labor) | NY Fed | Public chart series | Expectations + reservation wages |
| [NLSY79/97](https://www.bls.gov/nls/) | BLS | Cohort | Background + skills sub-analysis |

## Methods

<details>
<summary><strong>Estimation approaches</strong></summary>

- **Sequential OLS** — progressively adds control blocks (demographics, education,
  geography, job sorting, hours/schedule, family) to measure how much of the raw
  gap each block absorbs. This is the primary headline method.
- **Oaxaca-Blinder decomposition** — decomposes the gap into explained
  (observable-characteristic) and unexplained (coefficient-difference) components.
  Treated as supplemental due to post-2019 instability.
- **Double/debiased machine learning** — flexible nuisance-model adjustment using
  elastic net. Used as a sensitivity check; currently unweighted.
- **Employment-selection robustness** — inverse-probability weighting to separate
  worker wage gaps from total earnings gaps including non-workers.
- **Survey uncertainty** — successive-difference replication using 80 ACS replicate
  weights for all ACS analysis years.

</details>

<details>
<summary><strong>Sample definition</strong></summary>

Default sample: `prime_age_wage_salary`
- Age 25–54
- Wage/salary workers only
- Positive hours and positive pay
- Excludes active-duty military
- Excludes self-employed

</details>

<details>
<summary><strong>ACS control blocks (M0–M5)</strong></summary>

| Block | Controls added |
|-------|----------------|
| M0 | female only (raw gap) |
| M1 | + age, age², race/ethnicity, education |
| M2 | + state |
| M3 | + occupation, industry, class of worker |
| M4 | + usual hours, work from home, commute time |
| M5 | + marital status, children, children under 5 |

The biggest observed gap reduction comes from job sorting (M3): ~8 percentage
points in ACS 2023.

</details>

## Repo structure

```
scripts/          Analysis and download scripts
src/gender_gap/   Shared library (registry, settings, CLI)
data/             Raw and processed survey microdata (gitignored)
results/          CSV outputs, trend files, diagnostics
  trends/         Cross-year trend series
  diagnostics/    Robustness checks and validation artifacts
reports/          Narrative reports and technical notes
configs/          Dataset configuration and registry
crosswalks/       Geographic and classification crosswalks
tests/            Pytest suite
```

## Setup

Requires Python 3.11+.

```bash
pip install -e ".[dev]"
```

API keys for Census, BLS, BEA, and FRED go in `.env`. The download scripts pull
microdata from public APIs; no restricted-use files are needed.

```bash
# Download all source data
python scripts/download_all.py

# Run full analysis pipeline
python scripts/run_full_analysis.py

# Run tests
pytest
```

## License

This project is provided for research and educational use.
