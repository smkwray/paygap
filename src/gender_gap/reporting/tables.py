"""Report generation: tables, markdown summaries, CSV exports."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gender_gap.models.descriptive import gap_by_subgroup, raw_gap
from gender_gap.models.ols import OLSResult, results_to_dataframe

logger = logging.getLogger(__name__)


def export_raw_gap_table(
    df: pd.DataFrame,
    output_dir: Path,
    outcome: str = "hourly_wage_real",
) -> Path:
    """Export national raw gap table as CSV."""
    result = raw_gap(df, outcome)
    table = pd.DataFrame([result])
    path = output_dir / "national_raw_gap.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Exported raw gap table to %s", path)
    return path


def export_adjusted_gap_table(
    ols_results: list[OLSResult],
    output_dir: Path,
) -> Path:
    """Export OLS sequential model results as CSV."""
    table = results_to_dataframe(ols_results)
    path = output_dir / "national_adjusted_gap.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Exported adjusted gap table to %s", path)
    return path


def export_subgroup_tables(
    df: pd.DataFrame,
    group_cols: list[str],
    output_dir: Path,
    outcome: str = "hourly_wage_real",
) -> list[Path]:
    """Export subgroup gap tables as CSVs."""
    paths = []
    for col in group_cols:
        if col not in df.columns:
            continue
        table = gap_by_subgroup(df, col, outcome)
        path = output_dir / f"gap_by_{col}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        paths.append(path)
        logger.info("Exported subgroup gap table: %s", path)
    return paths


def export_markdown_summary(
    raw: dict,
    ols_results: list[OLSResult],
    output_dir: Path,
    oaxaca_result=None,
) -> Path:
    """Generate a markdown summary report."""
    lines = [
        "# Gender Earnings Gap Analysis — Summary Report\n",
        "## National Raw Gap\n",
        f"- Male mean hourly wage: ${raw['male_mean']:.2f}",
        f"- Female mean hourly wage: ${raw['female_mean']:.2f}",
        f"- Raw gap: ${raw['gap_dollars']:.2f} ({raw['gap_pct']:.1f}%)",
        f"- N male: {raw['n_male']:,} | N female: {raw['n_female']:,}",
        "",
        "## Sequential OLS Models (log hourly wage)\n",
        "| Model | Female Coef | SE | p-value | R² | N |",
        "|-------|-----------|----|---------|----|---|",
    ]
    for r in ols_results:
        lines.append(
            f"| {r.model_name} | {r.female_coef:.4f} | "
            f"{r.female_se:.4f} | {r.female_pvalue:.4f} | "
            f"{r.r_squared:.4f} | {r.n_obs:,} |"
        )

    if oaxaca_result is not None:
        lines.extend([
            "",
            "## Oaxaca-Blinder Decomposition\n",
            f"- Total log-wage gap: {oaxaca_result.total_gap:.4f}",
            f"- Explained (endowments): {oaxaca_result.explained:.4f} "
            f"({oaxaca_result.explained_pct:.1f}%)",
            f"- Unexplained (coefficients): {oaxaca_result.unexplained:.4f} "
            f"({oaxaca_result.unexplained_pct:.1f}%)",
        ])

    lines.extend([
        "",
        "## Methods and Assumptions\n",
        "- All estimates use survey weights.",
        "- Hourly wage = wage/salary income / (usual hours × weeks worked).",
        "- Dollar amounts deflated to 2024 base year.",
        "- Prime-age (25-54) wage-and-salary workers unless noted.",
        "- Public data identifies sex (male/female), not full gender identity.",
        "- A residual gap after rich controls is not identical to a causal "
        "estimate of discrimination.",
        "",
        "## Important: Multiple Model Families\n",
        "This analysis reports unadjusted, progressively adjusted, and "
        "mechanism-focused gaps. No single 'fully controlled' model is "
        "presented as the definitive answer.",
    ])

    path = output_dir / "summary_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("Exported markdown summary to %s", path)
    return path
