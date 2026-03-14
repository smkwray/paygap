"""Publication-quality charts for gender gap analysis.

All plots use matplotlib with a clean, minimal style suitable for
academic or policy publications.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Style defaults
COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "accent": "#059669",
    "neutral": "#6B7280",
    "ci": "#93C5FD",
}
FIG_DPI = 150
FIG_WIDTH = 8
FIG_HEIGHT = 5


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def plot_ols_sequential(
    ols_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Bar chart of female coefficient across sequential OLS models."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    models = ols_df["model"]
    coefs = ols_df["female_coef"]
    ses = ols_df["female_se"]

    x = np.arange(len(models))
    ax.bar(x, coefs, yerr=1.96 * ses, capsize=4,
                  color=COLORS["primary"], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Female coefficient (log wage)")
    ax.set_title("Gender wage gap: progressive controls")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved OLS sequential plot to %s", output_path)
    return output_path


def plot_quantile_coefficients(
    quantile_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Line plot of female coefficient across the wage distribution."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    q = quantile_df["quantile"]
    coef = quantile_df["female_coef"]
    se = quantile_df["female_se"]

    ax.fill_between(q, coef - 1.96 * se, coef + 1.96 * se,
                    alpha=0.25, color=COLORS["ci"])
    ax.plot(q, coef, marker="o", color=COLORS["primary"],
            linewidth=2, markersize=6, zorder=3)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Female coefficient (log wage)")
    ax.set_title("Gender gap across the wage distribution")
    ax.set_xticks(q)
    ax.set_xticklabels([f"{qi:.0%}" for qi in q])

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved quantile plot to %s", output_path)
    return output_path


def plot_heterogeneity_forest(
    het_df: pd.DataFrame,
    dimension: str,
    output_path: Path,
) -> Path:
    """Forest plot of gender gap by subgroup."""
    _setup_style()

    het_df = het_df.sort_values("gap")
    n_groups = len(het_df)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, max(4, n_groups * 0.5)))

    y = np.arange(n_groups)
    ax.errorbar(
        het_df["gap"], y,
        xerr=1.96 * het_df["se"],
        fmt="o", color=COLORS["primary"],
        capsize=4, markersize=6, linewidth=1.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(het_df["group"])
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Female coefficient (log wage)")
    ax.set_title(f"Gender gap by {dimension.replace('_', ' ')}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heterogeneity plot for %s to %s", dimension, output_path)
    return output_path


def plot_oaxaca_decomposition(
    oaxaca_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Stacked bar chart of Oaxaca-Blinder decomposition."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    # Expect columns: component, value
    # Or use the summary format
    if "explained" in oaxaca_df.columns:
        explained = oaxaca_df["explained"].iloc[0]
        unexplained = oaxaca_df["unexplained"].iloc[0]
    else:
        # Match component names case-insensitively (partial match)
        comp = oaxaca_df["component"].str.lower()
        explained = oaxaca_df.loc[comp.str.startswith("explained"), "value"].sum()
        unexplained = oaxaca_df.loc[comp.str.startswith("unexplained"), "value"].sum()

    total = explained + unexplained
    ax.barh(["Total gap"], [total], color=COLORS["neutral"], alpha=0.3, label="Total")
    ax.barh(["Explained"], [explained], color=COLORS["accent"], alpha=0.85, label="Explained")
    ax.barh(
        ["Unexplained"], [unexplained],
        color=COLORS["secondary"], alpha=0.85, label="Unexplained",
    )

    ax.set_xlabel("Log-wage gap")
    ax.set_title("Oaxaca-Blinder decomposition")

    # Add value labels
    for i, (label, val) in enumerate(
        [("Total gap", total), ("Explained", explained), ("Unexplained", unexplained)]
    ):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Oaxaca plot to %s", output_path)
    return output_path


def generate_all_plots(input_dir: Path, output_dir: Path) -> list[Path]:
    """Generate all available plots from model output CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    # OLS sequential
    ols_path = input_dir / "ols_sequential.csv"
    if ols_path.exists():
        df = pd.read_csv(ols_path)
        plots.append(plot_ols_sequential(df, output_dir / "ols_sequential.png"))

    # Quantile regression
    qr_path = input_dir / "quantile_regression.csv"
    if qr_path.exists():
        df = pd.read_csv(qr_path)
        plots.append(plot_quantile_coefficients(df, output_dir / "quantile_coefficients.png"))

    # Oaxaca decomposition
    oaxaca_path = input_dir / "oaxaca.csv"
    if oaxaca_path.exists():
        df = pd.read_csv(oaxaca_path)
        plots.append(plot_oaxaca_decomposition(df, output_dir / "oaxaca_decomposition.png"))

    # Heterogeneity plots (one per dimension)
    for het_path in sorted(input_dir.glob("heterogeneity_*.csv")):
        dim = het_path.stem.replace("heterogeneity_", "")
        df = pd.read_csv(het_path)
        if len(df) > 0:
            plots.append(plot_heterogeneity_forest(
                df, dim, output_dir / f"heterogeneity_{dim}.png"
            ))

    logger.info("Generated %d plots", len(plots))
    return plots
