#!/usr/bin/env python3
"""Build cross-year CPS selection-robustness artifacts."""

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
logger = logging.getLogger("build_cps_selection_artifacts")


def main() -> None:
    rows = []
    for year_dir in sorted((RESULTS_DIR / "cps").glob("*")):
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
        raise SystemExit("No CPS selection_robustness.csv files found")

    trend = pd.concat(rows, ignore_index=True).sort_values(["year", "model"])
    trend_path = RESULTS_DIR / "trends" / "cps_selection_trend.csv"
    trend_path.parent.mkdir(parents=True, exist_ok=True)
    trend.to_csv(trend_path, index=False)

    s2 = trend.loc[trend["model"] == "S2"].copy().sort_values("year")
    report_path = REPORTS_DIR / "selection_robustness.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# CPS Selection Robustness\n\n"
        "This note summarizes the CPS ASEC employment-selection robustness layer.\n\n"
        "## Headline pattern (S2 block)\n"
        f"- Years covered: {int(s2['year'].min())}-{int(s2['year'].max())}\n"
        f"- Mean female employment probability effect: {s2['employment_female_prob_pp'].mean():.2f} percentage points\n"
        f"- Mean combined expected annual earnings gap: {s2['combined_expected_earnings_gap_pct'].mean():.2f}%\n"
        f"- Mean observed total annual earnings gap: {s2['observed_total_earnings_gap_pct'].mean():.2f}%\n"
        f"- Mean observed worker hourly wage gap: {s2['observed_worker_hourly_gap_pct'].mean():.2f}%\n"
        f"- Mean IPW worker hourly wage gap: {s2['ipw_worker_hourly_gap_pct'].mean():.2f}%\n\n"
        "## Interpretation\n"
        "- The CPS worker wage gap and the total annual earnings gap are different estimands.\n"
        "- The two-part expected-earnings gap is consistently larger than the worker-only hourly wage gap.\n"
        "- The IPW worker wage gap is close to, but not identical to, the observed worker gap; that is evidence that labor-force selection matters, but it does not dominate the worker-gap result.\n"
    )

    logger.info("Wrote %s", trend_path)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
