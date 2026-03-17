# PSID Validation

This note uses the new 2023 PSID main-panel validation extract. It is a one-year cross-section built from the public family and cross-year individual files, limited to reference persons and spouses.

Important scope limits:
- this is not yet a multi-wave PSID panel analysis
- the file covers reference persons and spouses rather than the full household roster

- Descriptive hourly-wage gap: 24.86%
- Winsorized hourly-wage gap (p1-p99): 21.16%
- Raw annual-earnings gap: 36.41%
- Common-sample model raw hourly gap: 19.42%
- Final staged hourly gap: 15.01%
- Common complete-case sample: 7997
- Largest reduction block: state and job sorting (7.70 percentage points)

## Block Transitions

- state and job sorting: 22.61% -> 14.91% (7.70 points)
- work arrangement: 14.91% -> 14.80% (0.11 points)

## Oaxaca Snapshot

- Total gap: 100.00%
- Explained (endowments): 15.47%
- Unexplained (coefficients): 84.53%

## Reproductive-Stage Gaps

- recent_birth: 49.71% (69.75 vs 35.08)
- childless_other_partnered: 30.65% (46.17 vs 32.02)
- mother_6_17_only: 29.34% (43.91 vs 31.03)

## Takeaway

PSID now provides a live validation lane for the reproductive-burden extension. In its current form it is best read as a compact cross-check on the ACS story: job sorting, work arrangement, and family/reproductive variables all matter, but the lane is still narrower than ACS because its public-use demographic surface is coarser and it is not yet a full multi-wave panel treatment.
