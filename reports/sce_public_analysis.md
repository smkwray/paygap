# SCE Public-Series Empirical Note

This note uses the official New York Fed public chart-data workbook for the Labor Market Survey.
It provides an empirical SCE evidence layer without attempting a person-level merge onto ACS/CPS.

## Data used

- Chart-data workbook: `data/raw/sce_labor_market/sce_labor_chart_data_public.xlsx`
- Public microdata workbook staged for future work: `data/raw/sce_labor_market/sce_labor_microdata_public.xlsx`
- Latest public chart wave in the workbook: 2025-11

## Main findings

- Latest expected offer wage, women: 55.91
- Latest expected offer wage, men: 73.37
- Latest men-minus-women expected offer wage gap: 17.46
- Latest reservation wage, women: 69.78
- Latest reservation wage, men: 91.12
- Latest men-minus-women reservation wage gap: 21.34

- Average men-minus-women expected offer wage gap over the latest three waves: 16.30
- Average men-minus-women reservation wage gap over the latest three waves: 17.92

## Interpretation

- The official public SCE series show that men report higher expected offer wages and higher reservation wages than women in the latest available waves.
- That is directly relevant to the bargaining/expectations question: the difference shows up before any attempted person-level merge to ACS/CPS.
- This still does not become a control variable in the main wage regressions. It remains supporting evidence on expectations and outside options.

## Practical use in this repo

- Use ACS/CPS/ATUS as the main realized-gap evidence.
- Use this SCE public-series note to support claims about expected offers and reservation wages.
- If deeper SCE work is wanted later, the staged public microdata workbook can be used for more detailed within-SCE analysis if the needed subgroup variables are available or can be mapped cleanly.

## Latest series snapshot

| Date | Women expected offer | Men expected offer | Women reservation wage | Men reservation wage |
|---|---:|---:|---:|---:|
| 2024-03 | 57.23 | 82.05 | 66.27 | 95.48 |
| 2024-07 | 54.50 | 74.96 | 66.43 | 94.45 |
| 2024-11 | 60.51 | 70.78 | 70.18 | 92.39 |
| 2025-03 | 56.54 | 76.41 | 65.59 | 82.30 |
| 2025-07 | 60.50 | 72.09 | 74.37 | 90.09 |
| 2025-11 | 55.91 | 73.37 | 69.78 | 91.12 |

## Bottom line

The SCE public series now give this project a direct empirical expectations layer: women report lower reservation wages and lower expected offer wages than men in the latest official New York Fed public data. That supports a bargaining and outside-options channel, but it does not displace the realized worker-gap findings from ACS and CPS.
