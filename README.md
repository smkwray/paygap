# U.S. Gender Earnings Gap

**[Browse the interactive results site](https://smkwray.github.io/paygap/)**

`paygap` estimates the unadjusted and adjusted male/female hourly earnings gap in the United States from free federal survey microdata: the American Community Survey Public Use Microdata Sample (ACS PUMS, 2015–2023 without 2020), the Current Population Survey Annual Social and Economic Supplement (CPS ASEC, 2015–2023) and the Survey of Income and Program Participation (SIPP, 2023). Time-use, wage-expectation and cohort surveys add evidence on mechanisms.

In 2023 the raw hourly gap is 15–17% in all three surveys, and a residual gap of 10.9–17.0% remains after demographic, geographic, job-sorting, schedule and family controls. All estimates are descriptive: they show where the gap is statistically concentrated, not whether employers or workers cause it.

## Headline findings

| Survey (2023) | Raw hourly gap | Adjusted hourly gap |
|---|---:|---:|
| ACS | 16.81% | 13.23% |
| CPS ASEC | 15.77% | 16.96% |
| SIPP | 15.09% | 10.91% |

The ACS 2023 raw gap has a 90% confidence interval of 16.50–17.12%. Each survey's adjusted gap uses the controls that survey supports, so the three specifications are not identical.

- Across ACS years the raw gap ranges from 15.9% to 18.7%, while the adjusted gap stays between 13.2% and 14.2%.
- Job sorting (occupation, industry, class of worker) is the largest measured channel. In an order-invariant Gelbach decomposition across eight ACS years it accounts for about 70% of the explained gap, and reproductive-burden controls (recent birth, child age, couple type) for about 30%. Adding hours and schedule controls widens the gap.
- Double/debiased machine learning (DML) gives larger adjusted gaps than ordinary least squares (OLS) in all three surveys: 17.88% in ACS, 19.61% in CPS ASEC and 12.89% in SIPP.
- Reweighting for who is employed (inverse-probability weighting) does not shrink the worker hourly gap: in ACS, averaged over 2015–2023, it is 18.83% observed and 19.73% reweighted.
- In ACS 2023 quantile regressions the adjusted gap is 15.64% at the 10th percentile, 13.69% at the median and 17.00% at the 90th.
- In the American Time Use Survey (ATUS), women spend about 68 fewer minutes a day in paid work than men and about 43 more in housework and childcare.

Subgroup, occupation, variance, reproductive-burden, wage-expectation and cohort results are on the site and in [`reports/`](reports/README.md).

## Method

The default sample (`prime_age_wage_salary`) is wage and salary workers aged 25–54 with positive hours and pay, excluding active-duty military and the self-employed. The headline method is sequential OLS on log real hourly wages, adding blocks of controls in turn: demographics (age, race and ethnicity, education), state, job sorting, schedule (usual hours, work from home, commute time) and family (marital status, children). A Gelbach decomposition attributes the change in the female coefficient to each block regardless of order. DML with elastic-net nuisance models, an Oaxaca-Blinder decomposition and inverse-probability weighting for employment selection serve as checks. ACS standard errors use successive-difference replication. The [methods page](https://smkwray.github.io/paygap/methods.html) gives the full specification.

## Data sources

| Dataset | Agency | Years | Role |
|---|---|---|---|
| [ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html) | Census Bureau | 2015–2019, 2021–2023 | Primary wage estimates |
| [CPS ASEC](https://www.census.gov/programs-surveys/cps.html) | Census Bureau, Bureau of Labor Statistics (BLS) | 2015–2023 | Cross-check and employment selection |
| [SIPP](https://www.census.gov/programs-surveys/sipp.html) | Census Bureau | 2023 | Cross-check |
| [ATUS](https://www.bls.gov/tus/) | BLS | Pooled | Time use |
| [Survey of Consumer Expectations (SCE), labor market](https://www.newyorkfed.org/microeconomics/sce/labor) | Federal Reserve Bank of New York | Public chart series | Expected and reservation wages |
| [National Longitudinal Survey of Youth (NLSY79, NLSY97)](https://www.bls.gov/nls/) | BLS | Cohort | Skills and family-background controls |
| [O\*NET](https://www.onetcenter.org/) | Department of Labor | Version 30.2 | Job context (schedule rigidity, autonomy) |

The download script also fetches BLS and Bureau of Economic Analysis (BEA) area series (unemployment, prices, employment and regional price parities) used as contextual controls. Raw microdata are not committed; downloads land in `data/raw/`.

## Reproduce

Requires Python 3.11 or later. Set `CENSUS_API_KEY` and `BEA_API_KEY` (free keys from the Census Bureau and BEA) in your shell before downloading.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/download_all.py       # ACS, CPS ASEC, ATUS and area series into data/raw/
python scripts/run_all_analyses.py   # ACS, CPS ASEC and ATUS models into results/
pytest
```

`gender-gap download --dataset sipp` and `--dataset sce_labor_market` fetch SIPP and SCE. `scripts/run_gelbach_decomposition.py` and `scripts/run_repro_extension.py` run the Gelbach, reproductive-burden and variance extensions, and `scripts/build_*.py` write the reports. Two inputs are not downloaded: put the O\*NET 30.2 text files in `data/raw/context/onet/db_30_2_text/` and processed NLSY extracts in `data/external/nlsy/` (or the directory named by `NLSY_DATA_DIR`). If BLS refuses scripted downloads of its occupational wage files, the download script writes manual instructions to `data/raw/context/oews/DOWNLOAD_INSTRUCTIONS.md`.

To preview the results site locally, run `python3 -m http.server -d docs 8000` and open `http://localhost:8000`.

## Outputs and layout

```
results/          CSV outputs by survey (acs/, cps/, sipp/, atus/), plus trends/, gelbach/, repro/, variance/
reports/          Markdown reports, indexed in reports/README.md
docs/             Static results site
scripts/          Download, analysis and report scripts
src/gender_gap/   Python package and the gender-gap command
configs/          Dataset registry and extension settings
crosswalks/       Geographic and classification crosswalks
tests/            pytest suite
```

## Limitations

- The Census Bureau did not release standard 2020 ACS 1-year data, so the ACS series skips 2020.
- CPS ASEC and SIPP support fewer controls than ACS, and SIPP covers 2023 only.
- DML estimates do not use survey weights.
- Oaxaca-Blinder shares are unstable after 2019 (the unexplained share rises from about 74% to 88–94% while the total gap barely moves), so the decomposition is secondary to sequential OLS.
- The NLSY adult-resource controls (household income, net worth) are measured after labor-market entry, so they describe overlap with the gap rather than explain it.
- The variance extension pools ACS 2013–2024, a wider window than the headline 2015–2023 series.

## License

No open-source license is attached yet. The code and results are published for research and educational use.
