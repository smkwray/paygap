# ACS Oaxaca Review

## Headline

- Pre-2019 unexplained share average: 74.08%
- 2019+ unexplained share average: 87.92%
- Maximum unexplained share: 2021 at 93.62%
- Minimum unexplained share: 2015 at 73.53%

## Interpretation

- The unexplained share still jumps materially after 2019 even after the ACS family-field rebuild, but the ACS M5 adjusted gap remains in a comparatively tight band near 13% to 14%.
- The new sensitivity note in `reports/oaxaca_sensitivity.md` shows that this jump is driven more by the explained component collapsing than by the total wage gap blowing out.
- That combination suggests decomposition fragility is a live concern. The post-2019 movement should not be treated as the cleanest headline result even though it no longer looks like a mysterious end-to-end break.
- This does not look like an obvious end-to-end pipeline break because the raw gap and the sequential OLS trend remain broadly stable.
- The corrected ACS family fields modestly lowered the post-2019 unexplained shares, but not nearly enough to remove the jump.

## Year Table

| Year | Total log gap | Explained % | Unexplained % | M5 adjusted gap % |
|---|---:|---:|---:|---:|
| 2015 | 0.1794 | 26.47 | 73.53 | -14.20 |
| 2016 | 0.1731 | 25.48 | 74.52 | -13.65 |
| 2017 | 0.1708 | 26.13 | 73.87 | -13.65 |
| 2018 | 0.1702 | 25.62 | 74.38 | -13.59 |
| 2019 | 0.1692 | 18.23 | 81.77 | -13.86 |
| 2021 | 0.1639 | 6.38 | 93.62 | -14.21 |
| 2022 | 0.1654 | 11.52 | 88.48 | -13.76 |
| 2023 | 0.1623 | 12.19 | 87.81 | -13.23 |

Comparison artifact:

- `results/diagnostics/acs_family_rebuild_comparison.csv`
- `results/diagnostics/acs_oaxaca_sensitivity_summary.csv`
- `results/diagnostics/acs_oaxaca_sensitivity_yearly.csv`
- `reports/acs_family_rebuild_review.md`
- `reports/oaxaca_sensitivity.md`

## Recommended Follow-up

1. If you want to push Oaxaca further, recompute or benchmark it with broader control aggregation to test whether the explained component stabilizes.
2. Compare the post-2019 explained-component collapse against any future alternative sample restrictions or control sets.
3. Keep sequential OLS and raw-gap trends as the primary headline series; use Oaxaca as a supplemental decomposition result.
