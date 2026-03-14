# SIPP Models

This note adds a modest adjusted-gap surface for the validated 2023 public-use SIPP file.
The control sequence only uses variables that are actually present and well-covered in the current standardized SIPP output.

## Specification ladder

- `SIPP0`: female only
- `SIPP1`: + month
- `SIPP2`: + occupation and industry
- `SIPP3`: + usual hours, paid-hourly status, multiple-jobholder indicator

## Headline results

- Raw hourly wage gap: 15.09%
- Latest adjusted log-point gap (`SIPP3`): -0.1156
- Latest adjusted percent gap (`SIPP3`): -10.91%
- Latest adjusted model R² (`SIPP3`): 0.2530
- Worker observations in `SIPP3`: 182,658

## Interpretation

- This is a SIPP-specific adjustment surface, not a substitute for the richer ACS sequential models.
- The main value is to show whether the 2023 SIPP worker gap remains material after conditioning on month, job sorting, and basic job-structure variables.
- If SIPP is extended further, the next honest step is richer covariate recovery from the public-use release rather than simply adding more models on the same limited feature set.

## Model table

| Model | Female coef | Percent gap | R² | N |
|---|---:|---:|---:|---:|
| SIPP0 | -0.1448 | -13.48% | 0.0061 | 182,658 |
| SIPP1 | -0.1448 | -13.48% | 0.0066 | 182,658 |
| SIPP2 | -0.1050 | -9.97% | 0.2452 | 182,658 |
| SIPP3 | -0.1156 | -10.91% | 0.2530 | 182,658 |
