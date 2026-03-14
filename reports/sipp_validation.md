# SIPP Validation

This report documents the first real-data validation pass for the public-use Census SIPP standardization lane.

## Snapshot
- Rows: 437,168
- Calendar year: 2023 to 2023
- Female share: 51.42%
- Employed share: 46.91%
- Unemployed share: 1.73%
- Not in labor force share: 43.37%
- Missing labor-force status share: 14.55%

## Coverage
- Weekly earnings observed: 45.11%
- Hourly wage observed: 44.94%
- Usual hours observed: 45.11%
- Actual hours observed: 45.11%
- Occupation observed among employed: 96.04%
- Industry observed among employed: 96.04%
- Paid hourly among employed: 96.86%

## Weighted means among employed
- Weekly earnings: 1357.52
- Hourly wage: 37.82

## Interpretation
- The SIPP lane is now validated on a real public-use file rather than only synthetic tests.
- This is still a standardization and validation surface, not yet a full SIPP wage-model integration.
- If future SIPP releases change field names again, the main risk is alias coverage rather than file access.
