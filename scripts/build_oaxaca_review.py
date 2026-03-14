#!/usr/bin/env python3
"""Build a focused review note for the ACS Oaxaca trend shift."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    oax = pd.read_csv(RESULTS_DIR / "trends" / "acs_oaxaca_trend.csv")
    ols = pd.read_csv(RESULTS_DIR / "trends" / "acs_ols_trend.csv")
    m5 = ols.loc[ols["model"] == "M5", ["year", "pct_gap"]].copy()

    merged = oax.merge(m5, on="year", how="left")
    pre = merged.loc[merged["year"] <= 2018, "unexplained_pct"]
    post = merged.loc[merged["year"] >= 2019, "unexplained_pct"]
    max_row = merged.loc[merged["unexplained_pct"].idxmax()]
    min_row = merged.loc[merged["unexplained_pct"].idxmin()]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "oaxaca_review.md"
    lines = [
        "# ACS Oaxaca Review",
        "",
        "## Headline",
        "",
        f"- Pre-2019 unexplained share average: {pre.mean():.2f}%",
        f"- 2019+ unexplained share average: {post.mean():.2f}%",
        f"- Maximum unexplained share: {int(max_row['year'])} at {max_row['unexplained_pct']:.2f}%",
        f"- Minimum unexplained share: {int(min_row['year'])} at {min_row['unexplained_pct']:.2f}%",
        "",
        "## Interpretation",
        "",
        "- The unexplained share jumps materially after 2019, but the ACS M5 adjusted gap remains in a comparatively tight band near 13% to 14%.",
        "- That combination suggests decomposition fragility is a live concern. The post-2019 movement should not be treated as the cleanest headline result until it is stress-tested.",
        "- This does not look like an obvious end-to-end pipeline break because the raw gap and the sequential OLS trend remain broadly stable.",
        "",
        "## Year Table",
        "",
        "| Year | Total log gap | Explained % | Unexplained % | M5 adjusted gap % |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in merged.iterrows():
        lines.append(
            f"| {int(row['year'])} | {row['total_gap']:.4f} | {row['explained_pct']:.2f} | "
            f"{row['unexplained_pct']:.2f} | {row['pct_gap']:.2f} |"
        )

    lines.extend([
        "",
        "## Recommended Follow-up",
        "",
        "1. Recompute or benchmark Oaxaca with broader control aggregation to test sensitivity.",
        "2. Compare the post-2019 decomposition movement against the finished ACS SDR uncertainty layer once the all-year M5 backfill completes.",
        "3. Keep sequential OLS and raw-gap trends as the primary headline series until Oaxaca sensitivity work is complete.",
    ])

    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
