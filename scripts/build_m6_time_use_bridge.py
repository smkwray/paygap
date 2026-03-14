#!/usr/bin/env python3
"""Build an M6-style mechanism bridge using ATUS time-use evidence.

This is deliberately not a person-level ACS+ATUS merged regression.
It is a mechanism-aware summary layer that places ATUS care/work burdens
next to the completed ACS/CPS residual-gap and selection-robustness results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DIAG_DIR = RESULTS_DIR / "diagnostics"


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_summary_rows() -> pd.DataFrame:
    atus = _load_csv(RESULTS_DIR / "atus" / "time_use_by_gender.csv").set_index("activity")
    acs_uncertainty = _load_csv(DIAG_DIR / "acs_uncertainty_summary.csv")
    acs_selection = _load_csv(RESULTS_DIR / "trends" / "acs_selection_trend.csv")
    cps_selection = _load_csv(RESULTS_DIR / "trends" / "cps_selection_trend.csv")

    acs_m5 = acs_uncertainty.loc[acs_uncertainty["metric"] == "M5_female_coef"].copy()
    acs_s2 = acs_selection.loc[acs_selection["model"] == "S2"].copy()
    cps_s2 = cps_selection.loc[cps_selection["model"] == "S2"].copy()

    unpaid_gap = float(
        atus.loc["minutes_housework", "gap_minutes"]
        + atus.loc["minutes_childcare", "gap_minutes"]
    )
    net_workload_gap = float(
        atus.loc["minutes_paid_work_diary", "gap_minutes"]
        + atus.loc["minutes_housework", "gap_minutes"]
        + atus.loc["minutes_childcare", "gap_minutes"]
    )

    rows = [
        {
            "section": "atus",
            "metric": "paid_work_gap_minutes",
            "value": float(atus.loc["minutes_paid_work_diary", "gap_minutes"]),
            "note": "Female minus male gap in daily paid-work minutes.",
        },
        {
            "section": "atus",
            "metric": "housework_gap_minutes",
            "value": float(atus.loc["minutes_housework", "gap_minutes"]),
            "note": "Female minus male gap in daily housework minutes.",
        },
        {
            "section": "atus",
            "metric": "childcare_gap_minutes",
            "value": float(atus.loc["minutes_childcare", "gap_minutes"]),
            "note": "Female minus male gap in daily childcare minutes.",
        },
        {
            "section": "atus",
            "metric": "commute_gap_minutes",
            "value": float(atus.loc["minutes_commute_related_travel", "gap_minutes"]),
            "note": "Female minus male gap in daily commute-related travel minutes.",
        },
        {
            "section": "atus",
            "metric": "unpaid_work_gap_minutes",
            "value": unpaid_gap,
            "note": "Housework plus childcare female-minus-male daily gap.",
        },
        {
            "section": "atus",
            "metric": "net_paid_plus_unpaid_gap_minutes",
            "value": net_workload_gap,
            "note": "Paid work plus housework plus childcare female-minus-male daily gap.",
        },
        {
            "section": "acs",
            "metric": "m5_mean_female_coef",
            "value": float(acs_m5["estimate"].mean()),
            "note": "Average ACS M5 female coefficient over analysis years.",
        },
        {
            "section": "acs",
            "metric": "m5_latest_female_coef",
            "value": float(acs_m5.sort_values("year")["estimate"].iloc[-1]),
            "note": "Latest ACS M5 female coefficient.",
        },
        {
            "section": "acs",
            "metric": "s2_mean_ipw_worker_gap_pct",
            "value": float(acs_s2["ipw_worker_hourly_gap_pct"].mean()),
            "note": "Mean ACS S2 IPW worker hourly wage gap.",
        },
        {
            "section": "acs",
            "metric": "s2_mean_expected_earnings_gap_pct",
            "value": float(acs_s2["combined_expected_earnings_gap_pct"].mean()),
            "note": "Mean ACS S2 combined expected annual-earnings gap.",
        },
        {
            "section": "cps",
            "metric": "s2_mean_ipw_worker_gap_pct",
            "value": float(cps_s2["ipw_worker_hourly_gap_pct"].mean()),
            "note": "Mean CPS S2 IPW worker hourly wage gap.",
        },
        {
            "section": "cps",
            "metric": "s2_mean_expected_earnings_gap_pct",
            "value": float(cps_s2["combined_expected_earnings_gap_pct"].mean()),
            "note": "Mean CPS S2 combined expected annual-earnings gap.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, path: Path) -> Path:
    def metric(section: str, name: str) -> float:
        return float(summary.loc[(summary["section"] == section) & (summary["metric"] == name), "value"].iloc[0])

    lines = [
        "# M6 Time-Use Bridge",
        "",
        "This artifact is the defensible `M6` layer for the project.",
        "It is not a merged ACS+ATUS regression and should not be described that way.",
        "Instead, it uses ATUS as a separate mechanism module to interpret the residual ACS/CPS wage gaps after `M5` and the selection-robustness checks.",
        "",
        "## Why This Is Separate",
        "",
        "- ACS/CPS identify realized wages, hours, and worker characteristics at scale.",
        "- ATUS identifies daily time allocation, especially unpaid work and care burdens.",
        "- ATUS cannot be merged person-by-person onto ACS/CPS in this repo's public-data workflow.",
        "- So the right use is mechanism interpretation, not a literal control variable inside the wage regression.",
        "",
        "## ATUS Burden Snapshot",
        "",
        f"- Female paid-work time gap: {metric('atus', 'paid_work_gap_minutes'):.2f} minutes/day",
        f"- Female housework gap: {metric('atus', 'housework_gap_minutes'):.2f} minutes/day",
        f"- Female childcare gap: {metric('atus', 'childcare_gap_minutes'):.2f} minutes/day",
        f"- Female commute gap: {metric('atus', 'commute_gap_minutes'):.2f} minutes/day",
        f"- Female unpaid-work gap (housework + childcare): {metric('atus', 'unpaid_work_gap_minutes'):.2f} minutes/day",
        f"- Female net paid+unpaid gap: {metric('atus', 'net_paid_plus_unpaid_gap_minutes'):.2f} minutes/day",
        "",
        "## Residual Gap Context",
        "",
        f"- Mean ACS `M5` female coefficient: {metric('acs', 'm5_mean_female_coef'):.4f}",
        f"- Latest ACS `M5` female coefficient: {metric('acs', 'm5_latest_female_coef'):.4f}",
        f"- Mean ACS `S2` IPW worker hourly gap: {metric('acs', 's2_mean_ipw_worker_gap_pct'):.2f}%",
        f"- Mean ACS `S2` combined expected annual-earnings gap: {metric('acs', 's2_mean_expected_earnings_gap_pct'):.2f}%",
        f"- Mean CPS `S2` IPW worker hourly gap: {metric('cps', 's2_mean_ipw_worker_gap_pct'):.2f}%",
        f"- Mean CPS `S2` combined expected annual-earnings gap: {metric('cps', 's2_mean_expected_earnings_gap_pct'):.2f}%",
        "",
        "## Interpretation",
        "",
        "- The ATUS evidence is directionally consistent with a family/schedule mechanism: women spend materially less time in paid work and materially more time in unpaid household and childcare work.",
        "- That mechanism helps explain why annual-earnings gaps are larger than worker-only hourly wage gaps in the ACS/CPS selection surfaces.",
        "- But the ATUS burden differences do not make the residual ACS worker-gap disappear; they coexist with a still-material post-`M5` residual gap.",
        "",
        "## Use In The Final Project",
        "",
        "Present this as a mechanism-aware bridge after `M5` and the selection results:",
        "1. ACS/CPS establish the realized gap and residual worker-gap.",
        "2. Selection robustness distinguishes worker gaps from broader employment and annual-earnings gaps.",
        "3. ATUS explains one plausible channel: women carry more unpaid work and childcare time, which is consistent with schedule constraints and reduced paid-work time.",
        "4. SCE adds a separate expectations/reservation-wage channel.",
        "",
        "This is a stronger and more defensible story than pretending ATUS creates a literal merged `M6` wage regression.",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary_rows()
    summary.to_csv(DIAG_DIR / "m6_time_use_bridge_summary.csv", index=False)
    write_report(summary, REPORTS_DIR / "m6_time_use_bridge.md")
    print(f"Wrote {DIAG_DIR / 'm6_time_use_bridge_summary.csv'}")
    print(f"Wrote {REPORTS_DIR / 'm6_time_use_bridge.md'}")


if __name__ == "__main__":
    main()
