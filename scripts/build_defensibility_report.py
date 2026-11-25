#!/usr/bin/env python3
"""Build a defensibility-focused report and diagnostics from completed outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_RAW = PROJECT_ROOT / "data" / "raw"

sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _pct_from_log_coef(coef: float) -> float:
    return (np.exp(coef) - 1) * 100


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _acs_repweight_availability() -> dict[str, int]:
    """Inspect raw ACS files to see whether replicate weights are present."""
    out: dict[str, int] = {}
    paths = sorted((DATA_RAW / "acs").glob("acs_pums_*_api.parquet"))
    paths.extend(sorted((DATA_RAW / "acs").glob("acs_pums_*_api_repweights.parquet")))
    for path in paths:
        df = pd.read_parquet(path)
        out[path.name] = len([c for c in df.columns if c.startswith("PWGTP") and c != "PWGTP"])
    return out


def build_acs_diagnostic() -> pd.DataFrame:
    raw = _load_csv(RESULTS_DIR / "trends" / "acs_raw_gap_trend.csv")
    ols = _load_csv(RESULTS_DIR / "trends" / "acs_ols_trend.csv")
    oax = _load_csv(RESULTS_DIR / "trends" / "acs_oaxaca_trend.csv")

    m5 = (
        ols.loc[ols["model"] == "M5", ["year", "female_coef", "r_squared"]]
        .rename(columns={"female_coef": "m5_female_coef", "r_squared": "m5_r_squared"})
        .copy()
    )
    m5["m5_pct_gap"] = m5["m5_female_coef"].apply(_pct_from_log_coef)

    diag = (
        raw[["year", "gap_pct", "male_mean", "female_mean"]]
        .rename(columns={"gap_pct": "raw_gap_pct"})
        .merge(m5, on="year", how="left")
        .merge(
            oax[["year", "explained_pct", "unexplained_pct"]],
            on="year",
            how="left",
        )
        .sort_values("year")
    )

    diag["unexplained_pct_change"] = diag["unexplained_pct"].diff()
    diag["raw_gap_pct_change"] = diag["raw_gap_pct"].diff()
    diag["m5_pct_gap_change"] = diag["m5_pct_gap"].diff()
    return diag


def build_cps_diagnostic() -> pd.DataFrame:
    raw = _load_csv(RESULTS_DIR / "trends" / "cps_asec_raw_gap_trend.csv")
    ols = _load_csv(RESULTS_DIR / "trends" / "cps_asec_ols_trend.csv")

    full = (
        ols.loc[ols["model"] == "M_full", ["year", "female_coef", "r_squared"]]
        .rename(columns={"female_coef": "m_full_female_coef", "r_squared": "m_full_r_squared"})
        .copy()
    )
    full["m_full_pct_gap"] = full["m_full_female_coef"].apply(_pct_from_log_coef)

    diag = (
        raw[["year", "gap_pct", "male_mean", "female_mean"]]
        .rename(columns={"gap_pct": "raw_gap_pct"})
        .merge(full, on="year", how="left")
        .sort_values("year")
    )
    return diag


def build_artifact_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        rows.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "mtime": path.stat().st_mtime,
            "size_bytes": path.stat().st_size,
        })
    return pd.DataFrame(rows)


def write_report(
    acs_diag: pd.DataFrame,
    cps_diag: pd.DataFrame,
    repweight_counts: dict[str, int],
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pooled_raw = _load_csv(RESULTS_DIR / "acs_pooled" / "raw_gap_pooled.csv").iloc[0]
    pooled_ols = _load_csv(RESULTS_DIR / "acs_pooled" / "ols_pooled.csv")
    pooled_p5 = pooled_ols.loc[pooled_ols["model"] == "P5"].iloc[0]
    pooled_oax = _load_csv(RESULTS_DIR / "acs_pooled" / "oaxaca_pooled.csv")
    atus = _load_csv(RESULTS_DIR / "atus" / "time_use_by_gender.csv")
    cps_selection = _load_csv(RESULTS_DIR / "trends" / "cps_selection_trend.csv")
    acs_selection = _load_csv(RESULTS_DIR / "trends" / "acs_selection_trend.csv")

    latest_acs = acs_diag.iloc[-1]
    latest_cps = cps_diag.iloc[-1]
    max_oaxaca = acs_diag.loc[acs_diag["unexplained_pct"].idxmax()]
    cps_s2 = cps_selection.loc[cps_selection["model"] == "S2"]
    acs_s2 = acs_selection.loc[acs_selection["model"] == "S2"]
    uncertainty_path = RESULTS_DIR / "diagnostics" / "acs_uncertainty_summary.csv"
    uncertainty_df = _load_csv(uncertainty_path) if uncertainty_path.exists() else pd.DataFrame()

    repweight_lines = []
    for filename, count in repweight_counts.items():
        repweight_lines.append(f"- `{filename}`: {count} ACS replicate-weight columns present")

    repweight_positive = [name for name, count in repweight_counts.items() if count > 0]
    repweight_zero = [name for name, count in repweight_counts.items() if count == 0]
    repweight_backfill = [name for name in repweight_positive if name.endswith("_api_repweights.parquet")]
    if repweight_backfill and len(repweight_backfill) >= 8:
        repweight_status_lines = [
            "The original ACS API parquet files still contain only the main person weight (`PWGTP`), but the replicate-weight acquisition lane is now complete for the ACS analysis years.",
            "That means survey-consistent ACS uncertainty is now feasible from the current raw extracts.",
        ]
    elif repweight_positive and repweight_zero:
        repweight_status_lines = [
            "The original ACS API parquet files still contain only the main person weight (`PWGTP`), but a new replicate-weight acquisition lane is now partially populated.",
            "At least one richer ACS raw file now carries `PWGTP1`-`PWGTP80`, so survey-consistent ACS uncertainty is now feasible for the years that have been backfilled and still pending for the rest.",
        ]
    elif repweight_positive:
        repweight_status_lines = [
            "ACS raw parquet files now include replicate weights alongside the main person weight (`PWGTP`).",
            "That means survey-consistent ACS uncertainty is feasible from the current raw extracts.",
        ]
    else:
        repweight_status_lines = [
            "Current raw ACS parquet files contain the main person weight (`PWGTP`) but not the ACS replicate weights.",
            "That means publishable survey-design uncertainty for ACS headline estimates is not available from the current raw extracts and would require a redownload or different extract configuration.",
        ]

    uncertainty_lines: list[str] = []
    if not uncertainty_df.empty:
        latest_uncertainty_year = int(uncertainty_df["year"].max())
        latest_uncertainty = uncertainty_df.loc[uncertainty_df["year"] == latest_uncertainty_year]
        raw_row = latest_uncertainty.loc[latest_uncertainty["metric"] == "raw_gap_pct"]
        if not raw_row.empty:
            row = raw_row.iloc[0]
            uncertainty_lines.append(
                f"- ACS raw gap ({latest_uncertainty_year}) with SDR uncertainty: {row['estimate']:.2f}% "
                f"(SE {row['se']:.2f}, 90% CI {row['ci90_low']:.2f}% to {row['ci90_high']:.2f}%)"
            )
        m5_row = latest_uncertainty.loc[latest_uncertainty["metric"] == "M5_female_coef"]
        if not m5_row.empty:
            row = m5_row.iloc[0]
            uncertainty_lines.append(
                f"- ACS M5 female coefficient ({latest_uncertainty_year}) with SDR uncertainty: {row['estimate']:.4f} "
                f"(SE {row['se']:.4f}, 90% CI {row['ci90_low']:.4f} to {row['ci90_high']:.4f})"
            )
    raw_trend_path = RESULTS_DIR / "trends" / "acs_raw_gap_trend_with_uncertainty.csv"
    if raw_trend_path.exists():
        raw_trend = _load_csv(raw_trend_path)
        if not raw_trend.empty:
            avg_moe90 = raw_trend["moe90"].mean()
            uncertainty_lines.append(
                f"- ACS raw-gap uncertainty trend is complete for {len(raw_trend)} years; average 90% margin of error is {avg_moe90:.2f} percentage points"
            )

    lines = [
        "# Wage estimates and survey diagnostics",
        "",
        "## Headline Results",
        "",
        f"- ACS latest-year raw gap (2023): {latest_acs['raw_gap_pct']:.2f}%",
        (
            f"- ACS latest-year M5 adjusted gap (2023): {latest_acs['m5_female_coef']:.4f} "
            f"in log points, about {abs(latest_acs['m5_pct_gap']):.2f}% as an exponentiated percent gap"
        ),
        f"- ACS pooled raw gap: {pooled_raw['gap_pct']:.2f}%",
        (
            f"- ACS pooled P5 adjusted gap: {pooled_p5['female_coef']:.4f} in log points, "
            f"about {abs(_pct_from_log_coef(pooled_p5['female_coef'])):.2f}% as an exponentiated percent gap"
        ),
        f"- CPS latest-year raw gap (2023): {latest_cps['raw_gap_pct']:.2f}%",
        f"- CPS latest-year M_full adjusted gap (2023): {latest_cps['m_full_pct_gap']:.2f}%",
        "",
        "## Main Cautions",
        "",
        f"- The largest ACS Oaxaca unexplained share occurs in {int(max_oaxaca['year'])}: {max_oaxaca['unexplained_pct']:.2f}%.",
        "- Oaxaca unexplained shares rise sharply after 2019, while the M5 adjusted gaps stay comparatively stable. That makes the decomposition more fragile than the sequential OLS trend.",
        "- The pooled ACS model uses broad occupation and industry controls for tractability, so pooled and year-specific adjusted gaps should not be treated as identical estimands.",
        "- Correcting the ACS family fields modestly increased the year-by-year M5 gap in every year rather than attenuating it.",
        "- The SCE layer is implemented as a methodological supplement, but there is still no repo-local empirical SCE microdata analysis.",
        "",
        "## ACS Replicate-Weight Feasibility",
        "",
        *repweight_status_lines,
        "",
        *repweight_lines,
    ]

    if uncertainty_lines:
        lines.extend([
            "",
            "## ACS Uncertainty Snapshot",
            "",
            *uncertainty_lines,
        ])

    lines.extend([
        "",
        "## ATUS Mechanism Snapshot",
        "",
    ])

    for _, row in atus.iterrows():
        lines.append(
            f"- `{row['activity']}`: female-male difference = {row['gap_minutes']:.2f} minutes/day"
        )

    lines.extend([
        "",
        "## CPS Selection Snapshot",
        "",
        f"- CPS `S2` mean combined expected annual-earnings gap (2015-2023): {cps_s2['combined_expected_earnings_gap_pct'].mean():.2f}%",
        f"- CPS `S2` mean observed total annual-earnings gap (2015-2023): {cps_s2['observed_total_earnings_gap_pct'].mean():.2f}%",
        f"- CPS `S2` mean observed worker hourly wage gap (2015-2023): {cps_s2['observed_worker_hourly_gap_pct'].mean():.2f}%",
        f"- CPS `S2` mean IPW worker hourly wage gap (2015-2023): {cps_s2['ipw_worker_hourly_gap_pct'].mean():.2f}%",
        "",
        "## ACS Selection Snapshot",
        "",
        f"- ACS `S2` mean female employment probability effect (2015-2023): {acs_s2['employment_female_prob_pp'].mean():.2f} percentage points",
        f"- ACS `S2` mean combined expected annual-earnings gap (2015-2023): {acs_s2['combined_expected_earnings_gap_pct'].mean():.2f}%",
        f"- ACS `S2` mean observed total annual-earnings gap (2015-2023): {acs_s2['observed_total_earnings_gap_pct'].mean():.2f}%",
        f"- ACS `S2` mean observed worker hourly wage gap (2015-2023): {acs_s2['observed_worker_hourly_gap_pct'].mean():.2f}%",
        f"- ACS `S2` mean IPW worker hourly wage gap (2015-2023): {acs_s2['ipw_worker_hourly_gap_pct'].mean():.2f}%",
    ])

    path = REPORTS_DIR / "defensibility_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    diagnostics_dir = RESULTS_DIR / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    acs_diag = build_acs_diagnostic()
    cps_diag = build_cps_diagnostic()
    artifact_inventory = build_artifact_inventory()
    repweight_counts = _acs_repweight_availability()

    acs_diag.to_csv(diagnostics_dir / "acs_defensibility_diagnostic.csv", index=False)
    cps_diag.to_csv(diagnostics_dir / "cps_defensibility_diagnostic.csv", index=False)
    artifact_inventory.to_csv(diagnostics_dir / "artifact_inventory.csv", index=False)

    report_path = write_report(acs_diag, cps_diag, repweight_counts)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
