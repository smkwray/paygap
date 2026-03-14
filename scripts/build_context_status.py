#!/usr/bin/env python3
"""Summarize context-data acquisition status and staging guidance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CONTEXT = PROJECT_ROOT / "data" / "raw" / "context"
RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _rows() -> list[dict[str, object]]:
    qcew_years = sorted(
        int(path.stem.split("_")[-1]) for path in (RAW_CONTEXT / "qcew").glob("qcew_*.parquet")
    )
    oews_years = sorted(
        int(path.stem.split("_")[-1]) for path in (RAW_CONTEXT / "oews").glob("oews_*.parquet")
    )

    return [
        {
            "source": "LAUS",
            "status": "present" if (RAW_CONTEXT / "laus_states.parquet").exists() else "missing",
            "detail": "State unemployment parquet" if (RAW_CONTEXT / "laus_states.parquet").exists() else "No cached parquet in raw/context",
        },
        {
            "source": "CPI_U",
            "status": "present" if (RAW_CONTEXT / "cpi_u.parquet").exists() else "missing",
            "detail": "Annual CPI parquet" if (RAW_CONTEXT / "cpi_u.parquet").exists() else "No cached parquet in raw/context",
        },
        {
            "source": "BEA_RPP",
            "status": "present" if (RAW_CONTEXT / "bea_rpp.parquet").exists() else "missing",
            "detail": "Regional price parity parquet" if (RAW_CONTEXT / "bea_rpp.parquet").exists() else "No cached parquet in raw/context",
        },
        {
            "source": "QCEW",
            "status": "present" if qcew_years else "missing",
            "detail": f"Parquet years cached: {qcew_years}" if qcew_years else "No cached parquet years in raw/context/qcew",
        },
        {
            "source": "OEWS",
            "status": "staged_manual" if (RAW_CONTEXT / "oews" / "DOWNLOAD_INSTRUCTIONS.md").exists() else "missing",
            "detail": (
                f"Parsed parquet years cached: {oews_years}"
                if oews_years
                else "Manual staging required; see raw/context/oews/DOWNLOAD_INSTRUCTIONS.md"
            ),
        },
    ]


def build_status() -> pd.DataFrame:
    return pd.DataFrame(_rows())


def write_report(df: pd.DataFrame, path: Path) -> Path:
    lines = [
        "# Context Data Status",
        "",
        "This project uses LAUS, CPI-U, BEA RPP, QCEW, and OEWS as supplemental local-context inputs.",
        "The current acquisition state is:",
        "",
    ]

    for row in df.itertuples(index=False):
        lines.append(f"- `{row.source}`: {row.status} — {row.detail}")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- QCEW is the most realistic context-data automation target from this environment.",
        "- OEWS is currently blocked by BLS access controls for direct scripted download here, so it now has an explicit manual-staging path.",
        "- Context data remain supplemental and are not blocking the core ACS/CPS/ATUS findings surface.",
    ])

    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_status()
    csv_path = RESULTS_DIR / "context_data_status.csv"
    report_path = REPORTS_DIR / "context_data_status.md"
    df.to_csv(csv_path, index=False)
    write_report(df, report_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
