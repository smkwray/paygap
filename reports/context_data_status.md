# Context Data Status

This project uses LAUS, CPI-U, BEA RPP, QCEW, and OEWS as supplemental local-context inputs.
The current acquisition state is:

- `LAUS`: missing — No cached parquet in raw/context
- `CPI_U`: missing — No cached parquet in raw/context
- `BEA_RPP`: present — Regional price parity parquet
- `QCEW`: present — Parquet years cached: [2023]
- `OEWS`: staged_manual — Manual staging required; see raw/context/oews/DOWNLOAD_INSTRUCTIONS.md

## Interpretation

- QCEW is the most realistic context-data automation target from this environment.
- OEWS is currently blocked by BLS access controls for direct scripted download here, so it now has an explicit manual-staging path.
- Context data remain supplemental and are not blocking the core ACS/CPS/ATUS findings surface.
