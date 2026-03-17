"""Reporting helpers for the reproductive-burden extension."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from gender_gap.settings import (
    PROJECT_ROOT,
    SHARED_ALIASES_CATALOG,
    SHARED_DATASETS_CATALOG,
    shared_source_path,
)
from gender_gap.utils.yaml_compat import load_yaml

LOCAL_INVENTORY_CHECKS = {
    "paygap_root": ["README.md", "src", "scripts"],
    "dadgap_root": [],
    "sexg_root": [],
    "marage_root": [],
    "fp_gender_root": [],
    "nlsy79_path": [],
    "nlsy97_path": [],
    "psid_path": [],
    "ipums_acs_path": [],
    "existing_onet_path": [],
    "existing_atus_path": [],
    "existing_sipp_path": [],
}

ATUS_METRIC_LABELS = {
    "minutes_paid_work_diary": "paid work",
    "minutes_housework": "housework",
    "minutes_childcare": "childcare",
    "minutes_commute_related_travel": "commute-related travel",
    "minutes_eldercare": "eldercare",
    "minutes_work_at_home_diary": "work at home",
}

ATUS_STAGE_ORDER = [
    "overall",
    "mother_under6",
    "recent_birth",
    "mother_6_17_only",
    "childless_other_partnered",
    "childless_unpartnered",
]


def build_repro_inventory_usage(
    required_acs_years: list[int],
    onet_required_files: list[str],
) -> pd.DataFrame:
    """Summarize which repro-extension inputs are present, reused, or missing."""
    rows: list[dict] = []
    shared_aliases = _load_csv(SHARED_ALIASES_CATALOG)
    shared_datasets = _load_csv(SHARED_DATASETS_CATALOG)

    paygap_aliases = [
        row
        for row in shared_aliases
        if row.get("project") == "paygap"
        and any(token in row.get("legacy_path", "").lower() for token in ("acs", "onet", "sipp", "atus"))
    ]
    for row in paygap_aliases:
        rows.append(
            {
                "asset_group": "shared_alias",
                "asset_name": Path(row["legacy_path"]).name,
                "status": row.get("status", "unknown"),
                "legacy_path": row.get("legacy_path", ""),
                "canonical_path": row.get("canonical_path", ""),
                "note": "shared alias catalog",
            }
        )

    dataset_lookup = {row.get("canonical_path", ""): row for row in shared_datasets}
    for year in required_acs_years:
        filename = f"acs_pums_{year}_api_repweights.parquet"
        local = PROJECT_ROOT / "data" / "raw" / "acs" / filename
        shared = shared_source_path("census", "acs", "wave2", "paygap", "raw", "acs", filename)
        canonical = _canonical_for_legacy(paygap_aliases, str(local.relative_to(PROJECT_ROOT))) or (
            str(shared.relative_to(SHARED_ALIASES_CATALOG.parents[1])) if shared.exists() else ""
        )
        rows.append(
            {
                "asset_group": "acs",
                "asset_name": filename,
                "status": "present" if local.exists() or shared.exists() else "missing",
                "legacy_path": str(local.relative_to(PROJECT_ROOT)),
                "canonical_path": canonical,
                "note": "repro_extension",
            }
        )
    for filename in onet_required_files:
        local = PROJECT_ROOT / "data" / "raw" / "context" / "onet" / "db_30_2_text" / filename
        shared = shared_source_path("onet", "db_30_2_text", filename)
        canonical = ""
        for row in paygap_aliases:
            if row.get("legacy_path", "").endswith(filename):
                canonical = row.get("canonical_path", "")
                break
        if not canonical and shared.exists():
            canonical = str(shared.relative_to(SHARED_ALIASES_CATALOG.parents[1]))
        note = dataset_lookup.get(canonical, {}).get("source_url", "repro_extension")
        rows.append(
            {
                "asset_group": "onet",
                "asset_name": filename,
                "status": "present" if local.exists() or shared.exists() else "missing",
                "legacy_path": str(local.relative_to(PROJECT_ROOT)),
                "canonical_path": canonical,
                "note": note or "repro_extension",
            }
        )
    return pd.DataFrame(rows)


def build_optional_validation_status() -> pd.DataFrame:
    """Describe whether optional NLSY/PSID validation sources are actually usable."""
    shared_datasets = _load_csv(SHARED_DATASETS_CATALOG)
    canonical_paths = [row.get("canonical_path", "") for row in shared_datasets]
    canonical_names = {Path(path).name for path in canonical_paths if path}

    rows = []

    for filename in ["nlsy79_all_1979-2022.zip", "nlsy97_all_1997-2023.zip"]:
        raw_path = shared_source_path("misc", "large_payloads", "wave3c", "sexg", "raw", filename)
        processed_name = filename.replace("_all_1979-2022.zip", "_cfa_resid.csv").replace(
            "_all_1997-2023.zip", "_cfa_resid.csv"
        )
        processed_local = PROJECT_ROOT / "data" / "external" / "nlsy" / processed_name
        processed_shared = shared_source_path(
            "misc",
            "large_payloads",
            "wave4",
            "paygap",
            "processed",
            "nlsy",
            processed_name,
        )
        processed_catalog = processed_name in canonical_names
        if processed_local.exists() or processed_shared.exists() or processed_catalog:
            status = "ready"
            note = "Processed CFA residual file is available for current NLSY standardizer."
            canonical_path = str(processed_shared if processed_shared.exists() else raw_path)
        elif raw_path.exists():
            status = "raw_only"
            note = "Raw NLS Investigator export exists, but paygap still needs a variable-map/CFA preprocessing adapter."
            canonical_path = str(raw_path)
        else:
            status = "missing"
            note = "No raw or processed NLSY asset detected."
            canonical_path = ""
        rows.append(
            {
                "dataset": processed_name.split("_")[0].upper(),
                "status": status,
                "canonical_path": canonical_path,
                "expected_processed": str(processed_local.relative_to(PROJECT_ROOT)),
                "note": note,
            }
        )

    psid_public_dir = shared_source_path("umich", "psid_cds_tas", "public")
    psid_main_raw_dir = shared_source_path("umich", "psid", "main_public", "paygap", "raw", "psid")
    psid_processed_dir = shared_source_path("umich", "psid", "main_public", "paygap", "processed", "psid")
    psid_expected_processed = PROJECT_ROOT / "data" / "external" / "psid" / "psid_2023_analysis_ready.parquet"
    psid_processed_candidates = [
        path
        for path in canonical_paths
        if "psid" in path.lower()
        and "psid_cds_tas" not in path.lower()
        and "processed" in path.lower()
    ]
    if psid_processed_dir.exists() or psid_processed_candidates:
        psid_status = "ready"
        psid_note = "A processed main-panel PSID asset is present for paygap."
        psid_path = psid_processed_dir if psid_processed_dir.exists() else Path(psid_processed_candidates[0])
    elif psid_main_raw_dir.exists():
        psid_status = "raw_main_panel"
        psid_note = "Core PSID public-use main-panel ZIP files are present, but paygap still needs a preprocessing adapter."
        psid_path = psid_main_raw_dir
    elif psid_public_dir.exists():
        psid_status = "public_bundle_only"
        psid_note = "PSID CDS/TAS public bundles are present, but the main PSID panel is not analysis-ready for paygap."
        psid_path = psid_public_dir
    else:
        psid_status = "missing"
        psid_note = "No PSID public bundle or main panel asset detected."
        psid_path = Path()
    rows.append(
        {
            "dataset": "PSID",
            "status": psid_status,
            "canonical_path": str(psid_path) if str(psid_path) else "",
            "expected_processed": str(psid_expected_processed.relative_to(PROJECT_ROOT)),
            "note": psid_note,
        }
    )
    return pd.DataFrame(rows)


def write_repro_inventory_report(usage: pd.DataFrame, output_path: Path) -> Path:
    """Write a compact markdown report from the usage inventory."""
    present = usage.loc[usage["status"].isin(["present", "active"])]
    missing = usage.loc[usage["status"] == "missing"]
    lines = [
        "# Reproductive Extension Inventory Usage",
        "",
        f"- Tracked assets: {len(usage)}",
        f"- Present or active: {len(present)}",
        f"- Missing: {len(missing)}",
        "",
        "## Present / reused assets",
    ]
    for row in present.itertuples(index=False):
        lines.append(
            f"- `{row.asset_group}` / `{row.asset_name}`: {row.status} "
            f"({row.legacy_path} -> {row.canonical_path or 'local-only'})"
        )
    lines.append("")
    lines.append("## Missing assets")
    if missing.empty:
        lines.append("- None")
    else:
        for row in missing.itertuples(index=False):
            lines.append(f"- `{row.asset_group}` / `{row.asset_name}`: expected at `{row.legacy_path}`")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_local_inventory_report(inventory_path: Path) -> dict:
    """Check whether the configured sibling repos and local data paths exist."""
    resolved_inventory = inventory_path.expanduser().resolve()
    data = load_yaml(resolved_inventory.read_text(encoding="utf-8")) or {}
    checks = {
        key: _inspect_inventory_path(data.get(key), children, resolved_inventory.parent)
        for key, children in LOCAL_INVENTORY_CHECKS.items()
    }
    exists_count = sum(1 for item in checks.values() if item["exists"])
    configured_count = sum(1 for item in checks.values() if item["configured"])
    return {
        "inventory_path": str(resolved_inventory),
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "configured_paths": configured_count,
            "existing_paths": exists_count,
            "missing_paths": configured_count - exists_count,
        },
        "checks": checks,
    }


def write_local_inventory_report(report: dict, output_path: Path) -> Path:
    """Write local inventory checks to JSON and markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Local inventory report",
        "",
        f"- Inventory file: `{report['inventory_path']}`",
        f"- Generated at: `{report['generated_at']}`",
        f"- Configured paths: {report['summary']['configured_paths']}",
        f"- Existing paths: {report['summary']['existing_paths']}",
        f"- Missing paths: {report['summary']['missing_paths']}",
        "",
    ]
    for key, item in report["checks"].items():
        lines.append(f"## {key}")
        lines.append(f"- Configured: {item['configured']}")
        lines.append(f"- Exists: {item['exists']}")
        if item.get("path"):
            lines.append(f"- Path: `{item['path']}`")
        if item.get("missing_children"):
            lines.append(f"- Missing children: {', '.join(item['missing_children'])}")
        lines.append("")
    output_path.with_suffix(".md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def write_repro_summary(
    output_path: Path,
    available_years: list[int],
    inventory_usage: pd.DataFrame,
    missing_inputs: list[str],
    generated_files: list[Path],
    onet_coverage: pd.DataFrame | None = None,
    atus_status: str | None = None,
    sipp_status: str | None = None,
    optional_validation: pd.DataFrame | None = None,
) -> Path:
    """Write the narrative repro-extension summary."""
    validation_headlines = _build_validation_headlines()
    lines = [
        "# Reproductive-Burden Extension Summary",
        "",
        f"- ACS years available in this run: {', '.join(str(year) for year in available_years) if available_years else 'none'}",
        f"- Shared/local inventory rows tracked: {len(inventory_usage)}",
        f"- Missing required inputs: {len(missing_inputs)}",
        "",
        "## Missing inputs",
    ]
    if missing_inputs:
        lines.extend([f"- {item}" for item in missing_inputs])
    else:
        lines.append("- None")

    if onet_coverage is not None and not onet_coverage.empty:
        lines.extend(["", "## O*NET merge coverage"])
        for row in onet_coverage.itertuples(index=False):
            lines.append(
                f"- {row.survey_year}: {row.n_matched}/{row.n_obs} matched "
                f"({row.match_rate:.1%}) via `{row.merge_key}`"
            )

    lines.extend(["", "## Mechanism / robustness status"])
    lines.append(f"- ATUS: {atus_status or 'not run'}")
    lines.append(f"- SIPP: {sipp_status or 'not run'}")
    if optional_validation is not None and not optional_validation.empty:
        lines.extend(["", "## Optional validation data status"])
        for row in optional_validation.itertuples(index=False):
            lines.append(f"- {row.dataset}: {row.status}: {row.note}")
    if validation_headlines:
        lines.extend(["", "## Optional validation results"])
        lines.extend([f"- {line}" for line in validation_headlines])
    lines.extend(["", "## Generated files"])
    for path in generated_files:
        lines.append(f"- `{path}`")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_atus_mechanisms_report(mechanisms_path: Path, output_path: Path) -> Path:
    """Summarize the ATUS mechanism table in markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not mechanisms_path.exists():
        output_path.write_text("# ATUS Reproductive Mechanisms\n\n- ATUS mechanism table is missing.\n", encoding="utf-8")
        return output_path

    mechanisms = pd.read_csv(mechanisms_path)
    if mechanisms.empty:
        output_path.write_text("# ATUS Reproductive Mechanisms\n\n- ATUS mechanism table is empty.\n", encoding="utf-8")
        return output_path
    if "status" in mechanisms.columns and set(mechanisms["status"].dropna().astype(str)) == {"skipped"}:
        reason = mechanisms.get("reason", pd.Series(["unknown reason"])).iloc[0]
        output_path.write_text(
            "# ATUS Reproductive Mechanisms\n\n"
            f"- ATUS mechanism layer was skipped: {reason}\n",
            encoding="utf-8",
        )
        return output_path

    lines = [
        "# ATUS Reproductive Mechanisms",
        "",
        f"- Rows: {len(mechanisms)}",
        f"- Stages: {mechanisms['reproductive_stage'].nunique(dropna=True)}",
        f"- Metrics: {mechanisms['metric'].nunique(dropna=True)}",
        "",
        "## Overall contrasts",
    ]
    overall = mechanisms.loc[mechanisms["reproductive_stage"] == "overall"].copy()
    if overall.empty:
        lines.append("- No overall ATUS rows were available.")
    else:
        for metric in ["minutes_paid_work_diary", "minutes_housework", "minutes_childcare"]:
            row = _lookup_mechanism_row(overall, metric)
            if row is not None:
                lines.append(_format_atus_line("overall", row))

    lines.extend(["", "## Stage contrasts"])
    for stage in ATUS_STAGE_ORDER[1:]:
        stage_rows = mechanisms.loc[mechanisms["reproductive_stage"] == stage].copy()
        if stage_rows.empty:
            continue
        lines.append(f"### {stage}")
        for metric in ["minutes_childcare", "minutes_housework", "minutes_paid_work_diary"]:
            row = _lookup_mechanism_row(stage_rows, metric)
            if row is not None:
                lines.append(_format_atus_line(stage, row))
        lines.append("")

    extreme = _largest_atus_gap(mechanisms)
    if extreme is not None:
        lines.extend(
            [
                "## Largest absolute gap",
                "",
                (
                    f"- `{extreme['reproductive_stage']}` / `{ATUS_METRIC_LABELS.get(extreme['metric'], extreme['metric'])}`: "
                    f"{float(extreme['gap_minutes']):+.1f} minutes (women minus men)"
                ),
            ]
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def write_nlsy_validation_output(output_path: Path, source_path: Path | None = None) -> Path | None:
    """Promote the NLSY cohort-comparison diagnostic into the repro results surface."""
    source = source_path or (PROJECT_ROOT / "results" / "diagnostics" / "nlsy_cohort_comparison.csv")
    if not source.exists():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(source).to_csv(output_path, index=False)
    return output_path


def build_repro_release_manifest(
    output_paths: list[Path],
    inventory_usage: pd.DataFrame | None = None,
    optional_validation: pd.DataFrame | None = None,
) -> dict:
    """Create a machine-readable manifest for repro-extension outputs."""
    manifest = {
        "project": "paygap_reproductive_burden_extension",
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "outputs": [_artifact_metadata(path) for path in output_paths if path.exists()],
        "inventory_inputs": [],
        "optional_validation_status": [],
    }
    if inventory_usage is not None and not inventory_usage.empty:
        manifest["inventory_inputs"] = inventory_usage.fillna("").to_dict(orient="records")
    if optional_validation is not None and not optional_validation.empty:
        manifest["optional_validation_status"] = optional_validation.fillna("").to_dict(orient="records")
    return manifest


def write_repro_release_manifest(manifest: dict, output_path: Path) -> Path:
    """Write the repro release manifest to JSON and markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Repro Release Manifest",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Platform: `{manifest['platform']}`",
        f"- Output artifacts tracked: {len(manifest['outputs'])}",
        "",
        "## Output artifacts",
    ]
    for item in manifest["outputs"]:
        lines.append(
            f"- `{item['path']}`: {item['size_bytes']} bytes, sha256 `{item['sha256'][:12]}`, "
            f"rows={item.get('row_count', 'n/a')}"
        )
    if manifest.get("inventory_inputs"):
        lines.extend(["", "## Inventory inputs"])
        for row in manifest["inventory_inputs"]:
            lines.append(f"- `{row.get('asset_name', '')}`: {row.get('status', '')}")
    if manifest.get("optional_validation_status"):
        lines.extend(["", "## Optional validation status"])
        for row in manifest["optional_validation_status"]:
            lines.append(f"- `{row.get('dataset', '')}`: {row.get('status', '')}")
    output_path.with_suffix(".md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def validate_repro_output_schemas(schema_path: Path) -> dict:
    """Compare current repro output columns against the tracked schema snapshot."""
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    matches = []
    mismatches = []
    missing = []
    for rel_path, expected_columns in files.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        actual_columns = list(pd.read_csv(path, nrows=0).columns)
        if actual_columns == expected_columns:
            matches.append(rel_path)
        else:
            mismatches.append(
                {
                    "path": rel_path,
                    "expected_columns": expected_columns,
                    "actual_columns": actual_columns,
                }
            )
    return {
        "schema_path": str(schema_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "checked_files": len(files),
        "matched_files": len(matches),
        "missing_files": missing,
        "mismatched_files": mismatches,
        "passed": not missing and not mismatches,
    }


def write_repro_schema_check(report: dict, output_path: Path) -> Path:
    """Write repro schema validation results to JSON and markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Repro Schema Check",
        "",
        f"- Schema snapshot: `{report['schema_path']}`",
        f"- Generated at: `{report['generated_at']}`",
        f"- Checked files: {report['checked_files']}",
        f"- Matched files: {report['matched_files']}",
        f"- Passed: {report['passed']}",
        "",
        "## Missing files",
    ]
    if report["missing_files"]:
        lines.extend([f"- `{path}`" for path in report["missing_files"]])
    else:
        lines.append("- None")
    lines.extend(["", "## Schema mismatches"])
    if report["mismatched_files"]:
        for item in report["mismatched_files"]:
            lines.append(f"- `{item['path']}`")
            lines.append(f"  expected: {item['expected_columns']}")
            lines.append(f"  actual:   {item['actual_columns']}")
    else:
        lines.append("- None")
    output_path.with_suffix(".md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _canonical_for_legacy(rows: list[dict], legacy_path: str) -> str:
    for row in rows:
        if row.get("legacy_path") == f"paygap/{legacy_path}":
            return row.get("canonical_path", "")
    return ""


def _build_validation_headlines() -> list[str]:
    diagnostics_dir = PROJECT_ROOT / "results" / "diagnostics"
    lines: list[str] = []

    nlsy_path = diagnostics_dir / "nlsy_cohort_comparison.csv"
    if nlsy_path.exists():
        nlsy = pd.read_csv(nlsy_path)
        for row in nlsy.itertuples(index=False):
            lines.append(
                f"{row.dataset}: raw gap {float(row.raw_gap_pct):.2f}%, final gap {float(row.final_gap_pct):.2f}%, "
                f"largest reduction {row.largest_reduction_block} ({float(row.largest_reduction_pp):.2f} pp)"
            )

    psid_summary_path = diagnostics_dir / "psid_validation_summary.csv"
    if psid_summary_path.exists():
        psid_metrics = pd.read_csv(psid_summary_path)
        metric_lookup = psid_metrics.set_index("metric")["value"].to_dict()
        if "descriptive_hourly_gap_pct" in metric_lookup and "final_hourly_gap_pct" in metric_lookup:
            winsorized = metric_lookup.get("winsorized_hourly_gap_pct")
            winsorized_text = (
                f", winsorized hourly gap {float(winsorized):.2f}%"
                if winsorized is not None and pd.notna(winsorized)
                else ""
            )
            lines.append(
                f"PSID 2023: descriptive hourly gap {float(metric_lookup['descriptive_hourly_gap_pct']):.2f}%"
                f"{winsorized_text}, final staged gap {float(metric_lookup['final_hourly_gap_pct']):.2f}%"
            )

    psid_panel_path = diagnostics_dir / "psid_panel_trend_summary.csv"
    if psid_panel_path.exists():
        panel = pd.read_csv(psid_panel_path)
        if not panel.empty and {"survey_year", "final_hourly_gap_pct"}.issubset(panel.columns):
            panel = panel.sort_values("survey_year")
            first = panel.iloc[0]
            last = panel.iloc[-1]
            winsorized_text = (
                f", winsorized hourly {float(first['winsorized_hourly_gap_pct']):.2f}% -> "
                f"{float(last['winsorized_hourly_gap_pct']):.2f}%"
                if "winsorized_hourly_gap_pct" in panel.columns
                else ""
            )
            if int(first["survey_year"]) != int(last["survey_year"]):
                lines.append(
                    f"PSID panel {int(first['survey_year'])}-{int(last['survey_year'])}: final staged gap "
                    f"{float(first['final_hourly_gap_pct']):.2f}% -> {float(last['final_hourly_gap_pct']):.2f}%"
                    f"{winsorized_text}"
                )

    return lines


def _inspect_inventory_path(path_str: str | None, required_children: list[str], base_dir: Path) -> dict:
    if not path_str:
        return {"configured": False, "exists": False, "missing_children": required_children}
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    exists = candidate.exists()
    missing_children = []
    if exists:
        for child in required_children:
            if not (candidate / child).exists():
                missing_children.append(child)
    return {
        "configured": True,
        "path": str(candidate),
        "exists": exists,
        "missing_children": missing_children,
    }


def _lookup_mechanism_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    match = df.loc[df["metric"] == metric]
    if match.empty:
        return None
    return match.iloc[0]


def _format_atus_line(stage: str, row: pd.Series) -> str:
    metric = ATUS_METRIC_LABELS.get(str(row["metric"]), str(row["metric"]))
    female_mean = row.get("female_mean_minutes")
    male_mean = row.get("male_mean_minutes")
    gap = row.get("gap_minutes")
    n_male = int(row.get("n_male", 0) or 0)
    if pd.isna(male_mean) or n_male == 0:
        return f"- `{metric}`: women average {float(female_mean):.1f} minutes; male comparison sample is unavailable."
    return (
        f"- `{metric}`: women minus men = {float(gap):+.1f} minutes "
        f"(women {float(female_mean):.1f}, men {float(male_mean):.1f})."
    )


def _largest_atus_gap(mechanisms: pd.DataFrame) -> pd.Series | None:
    numeric_gap = pd.to_numeric(mechanisms.get("gap_minutes"), errors="coerce")
    if numeric_gap.notna().sum() == 0:
        return None
    idx = numeric_gap.abs().idxmax()
    return mechanisms.loc[idx]


def _artifact_metadata(path: Path) -> dict:
    rel = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    item = {
        "path": rel,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "suffix": path.suffix.lower(),
    }
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        item["row_count"] = int(len(frame))
        item["columns"] = list(frame.columns)
    return item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
