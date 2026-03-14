#!/usr/bin/env python3
"""Build cross-year ACS selection-robustness artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("build_acs_selection_artifacts")


def main() -> None:
    rows = []
    for year_dir in sorted((RESULTS_DIR / "acs").glob("*")):
        if not year_dir.is_dir():
            continue
        path = year_dir / "selection_robustness.csv"
        if not path.exists():
            continue
        year = int(year_dir.name)
        df = pd.read_csv(path)
        df.insert(0, "year", year)
        rows.append(df)

    if not rows:
        raise SystemExit("No ACS selection_robustness.csv files found")

    trend = pd.concat(rows, ignore_index=True).sort_values(["year", "model"])
    trend_path = RESULTS_DIR / "trends" / "acs_selection_trend.csv"
    trend_path.parent.mkdir(parents=True, exist_ok=True)
    trend.to_csv(trend_path, index=False)

    s2 = trend.loc[trend["model"] == "S2"].copy().sort_values("year")
    report_path = REPORTS_DIR / "acs_selection_robustness.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# ACS Selection Robustness\n\n"
        "This note summarizes the ACS employment-selection robustness layer.\n\n"
        "## Headline pattern (S2 block)\n"
        f"- Years covered: {int(s2['year'].min())}-{int(s2['year'].max())}\n"
        f"- Mean female employment probability effect: {s2['employment_female_prob_pp'].mean():.2f} percentage points\n"
        f"- Mean combined expected annual earnings gap: {s2['combined_expected_earnings_gap_pct'].mean():.2f}%\n"
        f"- Mean observed total annual earnings gap: {s2['observed_total_earnings_gap_pct'].mean():.2f}%\n"
        f"- Mean observed worker hourly wage gap: {s2['observed_worker_hourly_gap_pct'].mean():.2f}%\n"
        f"- Mean IPW worker hourly wage gap: {s2['ipw_worker_hourly_gap_pct'].mean():.2f}%\n\n"
        "## Interpretation\n"
        "- This is a robustness layer for labor-force selection, not a full structural selection correction.\n"
        "- The worker-only hourly wage gap and the combined expected annual-earnings gap answer different questions.\n"
        "- The IPW worker wage gap should be read as a sensitivity check on worker-only estimates, not as the main estimand.\n"
    )

    logger.info("Wrote %s", trend_path)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
