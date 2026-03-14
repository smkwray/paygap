#!/usr/bin/env python3
"""Build consolidated multi-year trend charts for the final deliverable."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TRENDS_DIR = RESULTS_DIR / "trends"
PLOTS_DIR = TRENDS_DIR / "plots"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("trend_charts")

COLORS = {
    "acs_raw": "#0F766E",
    "acs_adj": "#1D4ED8",
    "cps_raw": "#A16207",
    "cps_adj": "#B91C1C",
    "oaxaca": "#7C3AED",
    "band": "#99F6E4",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def build_acs_gap_chart() -> Path:
    raw = pd.read_csv(TRENDS_DIR / "acs_raw_gap_trend_with_uncertainty.csv")
    m5 = pd.read_csv(TRENDS_DIR / "acs_ols_trend.csv")
    m5 = m5.loc[m5["model"] == "M5", ["year", "pct_gap"]].copy()

    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(
        raw["year"],
        raw["ci90_low"],
        raw["ci90_high"],
        color=COLORS["band"],
        alpha=0.55,
        label="ACS raw gap 90% CI",
    )
    ax.plot(raw["year"], raw["gap_pct"], marker="o", linewidth=2.2, color=COLORS["acs_raw"], label="ACS raw gap")
    ax.plot(m5["year"], m5["pct_gap"].abs(), marker="s", linewidth=2.2, color=COLORS["acs_adj"], label="ACS M5 adjusted gap")
    ax.set_title("ACS Gender Wage Gap Trend")
    ax.set_ylabel("Percent gap")
    ax.set_xlabel("Year")
    ax.legend(frameon=False)
    fig.tight_layout()

    out = PLOTS_DIR / "acs_gap_trend.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def build_cross_dataset_chart() -> Path:
    acs = pd.read_csv(TRENDS_DIR / "acs_raw_gap_trend.csv")[["year", "gap_pct"]].rename(columns={"gap_pct": "acs_raw_gap"})
    cps_raw = pd.read_csv(TRENDS_DIR / "cps_asec_raw_gap_trend.csv")[["year", "gap_pct"]].rename(columns={"gap_pct": "cps_raw_gap"})
    cps_adj = pd.read_csv(TRENDS_DIR / "cps_asec_ols_trend.csv")
    cps_adj = cps_adj.loc[cps_adj["model"] == "M_full", ["year", "pct_gap"]].rename(columns={"pct_gap": "cps_adj_gap"})

    merged = acs.merge(cps_raw, on="year", how="inner").merge(cps_adj, on="year", how="inner")

    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(merged["year"], merged["acs_raw_gap"], marker="o", linewidth=2.0, color=COLORS["acs_raw"], label="ACS raw gap")
    ax.plot(merged["year"], merged["cps_raw_gap"], marker="^", linewidth=2.0, color=COLORS["cps_raw"], label="CPS raw gap")
    ax.plot(merged["year"], merged["cps_adj_gap"].abs(), marker="d", linewidth=2.0, color=COLORS["cps_adj"], label="CPS adjusted gap")
    ax.set_title("ACS and CPS Gap Comparison")
    ax.set_ylabel("Percent gap")
    ax.set_xlabel("Year")
    ax.legend(frameon=False)
    fig.tight_layout()

    out = PLOTS_DIR / "cross_dataset_gap_trend.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def build_oaxaca_chart() -> Path:
    df = pd.read_csv(TRENDS_DIR / "acs_oaxaca_trend.csv")

    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["year"], df["unexplained_pct"], marker="o", linewidth=2.2, color=COLORS["oaxaca"], label="Unexplained share")
    ax.plot(df["year"], df["explained_pct"], marker="s", linewidth=1.8, color=COLORS["neutral"] if "neutral" in COLORS else "#6B7280", label="Explained share")
    ax.set_title("ACS Oaxaca Decomposition Trend")
    ax.set_ylabel("Percent of total gap")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = PLOTS_DIR / "acs_oaxaca_trend.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_acs_gap_chart(),
        build_cross_dataset_chart(),
        build_oaxaca_chart(),
    ]
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
