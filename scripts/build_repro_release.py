#!/usr/bin/env python3
"""Build lightweight release diagnostics for the repro extension from existing outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gender_gap.reporting.repro import (
    build_local_inventory_report,
    build_optional_validation_status,
    build_repro_inventory_usage,
    build_repro_release_manifest,
    validate_repro_output_schemas,
    write_atus_mechanisms_report,
    write_local_inventory_report,
    write_nlsy_validation_output,
    write_repro_release_manifest,
    write_repro_schema_check,
)
from gender_gap.settings import PROJECT_ROOT, load_repro_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "repro_extension.yaml",
        help="Repro extension config path.",
    )
    args = parser.parse_args()

    cfg = load_repro_config(args.config)
    paths = cfg.get("paths", {})
    acs_years = cfg.get("datasets", {}).get("acs", {}).get("years", [])
    required_onet = cfg.get("datasets", {}).get("onet", {}).get("required_files", [])

    inventory_usage_path = PROJECT_ROOT / paths.get("inventory_usage_output", "results/diagnostics/repro_inventory_usage.csv")
    optional_validation_path = PROJECT_ROOT / paths.get(
        "optional_validation_output",
        "results/diagnostics/repro_optional_validation_status.csv",
    )
    inventory_config = PROJECT_ROOT / paths.get("inventory_config", "inventory/inventory_paths.yaml")
    local_inventory_output = PROJECT_ROOT / paths.get("local_inventory_output", "diagnostics/local_inventory_report.json")
    atus_report_output = PROJECT_ROOT / paths.get("atus_report_output", "reports/atus_repro_mechanisms.md")
    atus_mechanisms_path = PROJECT_ROOT / paths.get("results_dir", "results/repro") / "atus_mechanisms.csv"
    nlsy_validation_output = PROJECT_ROOT / paths.get("nlsy_validation_output", "results/repro/nlsy_validation.csv")
    release_manifest_output = PROJECT_ROOT / paths.get("release_manifest_output", "diagnostics/repro_release_manifest.json")
    schema_snapshot_path = PROJECT_ROOT / paths.get("schema_snapshot_path", "configs/repro_output_schemas.json")
    schema_check_output = PROJECT_ROOT / paths.get("schema_check_output", "diagnostics/repro_schema_check.json")

    if inventory_usage_path.exists():
        inventory_usage = pd.read_csv(inventory_usage_path)
    else:
        inventory_usage = build_repro_inventory_usage(acs_years, required_onet)
        inventory_usage_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_usage.to_csv(inventory_usage_path, index=False)

    if optional_validation_path.exists():
        optional_validation = pd.read_csv(optional_validation_path)
    else:
        optional_validation = build_optional_validation_status()
        optional_validation_path.parent.mkdir(parents=True, exist_ok=True)
        optional_validation.to_csv(optional_validation_path, index=False)

    if inventory_config.exists():
        report = build_local_inventory_report(inventory_config)
        write_local_inventory_report(report, local_inventory_output)

    if atus_mechanisms_path.exists():
        write_atus_mechanisms_report(atus_mechanisms_path, atus_report_output)

    write_nlsy_validation_output(nlsy_validation_output)

    output_paths = sorted((PROJECT_ROOT / paths.get("results_dir", "results/repro")).glob("*.csv"))
    output_paths.extend(
        path
        for path in [
            PROJECT_ROOT / paths.get("report_path", "reports/repro_extension_summary.md"),
            atus_report_output,
            local_inventory_output,
            local_inventory_output.with_suffix(".md"),
        ]
        if path.exists()
    )
    manifest = build_repro_release_manifest(
        output_paths=output_paths,
        inventory_usage=inventory_usage,
        optional_validation=optional_validation,
    )
    write_repro_release_manifest(manifest, release_manifest_output)

    if schema_snapshot_path.exists():
        schema_report = validate_repro_output_schemas(schema_snapshot_path)
        write_repro_schema_check(schema_report, schema_check_output)

    print(release_manifest_output)


if __name__ == "__main__":
    main()
