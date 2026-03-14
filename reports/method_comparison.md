# DML vs OLS vs Oaxaca

This note compares three different gap-estimation lenses on the latest available public-data files.

Important caution: these are not identical estimands.
- OLS reports a conditional female coefficient under a specified control set.
- DML reports a residual female effect after flexible nuisance-model adjustment.
- Oaxaca reports an explained/unexplained decomposition under a chosen reference structure.

## 2023 comparison

| Dataset | Raw gap % | OLS model | OLS adjusted gap % | DML adjusted gap % | Oaxaca unexplained % |
|---|---:|---|---:|---:|---:|
| ACS | 16.81 | M5 | 13.23 | 17.88 | 87.81 |
| CPS ASEC | 15.77 | M_full | 16.96 | 19.61 | 90.88 |
| SIPP | 15.09 | SIPP3 | 10.91 | 12.89 | 85.60 |

## Interpretation

- OLS and DML are the more comparable pair for an adjusted residual gap; Oaxaca answers a different question.
- In this 2023 comparison, DML runs larger than OLS in ACS, CPS, and SIPP, so the flexible residualization here does not drive the female effect toward zero.
- If Oaxaca behaves differently, especially through unstable explained shares, that should be treated as a decomposition caution rather than a contradiction of the residual-gap estimates.

## Notes

- ACS uses the canonical 2023 Oaxaca table already in the repo.
- CPS and SIPP Oaxaca values in this note are ad hoc 2023 decompositions computed with simple numeric controls because those datasets do not already have canonical Oaxaca artifacts.
- DML here uses the elastic-net nuisance learner for computational tractability and comparability across datasets.
- The current DML implementation is unweighted, so this comparison should be treated as a supplemental sensitivity layer rather than a survey-primary estimate.
