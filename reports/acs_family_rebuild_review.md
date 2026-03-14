# ACS Family Rebuild Review

## Purpose

This note compares the final ACS outputs rebuilt from corrected family fields (`NOC`, `PAOC`) against the earlier placeholder-family ACS surface where `number_children` and `children_under_5` were effectively zeros.

Comparison artifact:

- `results/diagnostics/acs_family_rebuild_comparison.csv`

## Headline finding

Correcting the ACS family fields did not attenuate the residual ACS `M5` gap. It made the year-by-year `M5` female coefficient modestly more negative in every year.

## Year-by-year M5 change

| Year | Old M5 | Rebuilt M5 | Delta |
|---|---:|---:|---:|
| 2015 | -0.1486 | -0.1532 | -0.0046 |
| 2016 | -0.1423 | -0.1468 | -0.0045 |
| 2017 | -0.1425 | -0.1467 | -0.0043 |
| 2018 | -0.1432 | -0.1461 | -0.0028 |
| 2019 | -0.1454 | -0.1493 | -0.0038 |
| 2021 | -0.1501 | -0.1533 | -0.0032 |
| 2022 | -0.1468 | -0.1481 | -0.0013 |
| 2023 | -0.1408 | -0.1419 | -0.0012 |

## Pooled ACS change

- Old pooled `P5`: `-0.2068`
- Rebuilt pooled `P5`: `-0.2110`
- Delta: `-0.0042`

## Oaxaca implication

The family-field rebuild does not eliminate the post-2019 Oaxaca jump.

- 2019 unexplained share: `83.03%` -> `81.77%`
- 2021 unexplained share: `96.48%` -> `93.62%`
- 2022 unexplained share: `91.69%` -> `88.48%`
- 2023 unexplained share: `90.19%` -> `87.81%`

That means the Oaxaca concern remains real, but it is not primarily a byproduct of the earlier ACS family-field placeholder bug.

## Interpretation

- The corrected family fields modestly improve specification realism.
- They do not explain away the residual ACS `M5` gap.
- They also do not explain away the post-2019 Oaxaca instability.
- Sequential OLS remains the cleaner headline series; Oaxaca still needs careful sensitivity treatment.
