"""Reporting helpers for the dedicated variance addon surface."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from gender_gap.settings import PROJECT_ROOT


def write_variance_inventory_report(usage: pd.DataFrame, output_path: Path) -> Path:
    """Write a compact markdown inventory summary for the variance addon."""
    present = usage.loc[usage["status"].isin(["present", "active", "ready"])]
    missing = usage.loc[usage["status"] == "missing"]
    lines = [
        "# Variance Addon Inventory Usage",
        "",
        f"- Tracked assets: {len(usage)}",
        f"- Present or ready: {len(present)}",
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
            lines.append(
                f"- `{row.asset_group}` / `{row.asset_name}`: expected at `{row.legacy_path}`"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_variance_summary(
    output_path: Path,
    available_years: list[int],
    inventory_usage: pd.DataFrame,
    reused_outputs: list[Path],
    addon_outputs: list[Path],
    missing_inputs: list[str],
    optional_validation: pd.DataFrame | None = None,
    onet_coverage: pd.DataFrame | None = None,
    atus_status: str | None = None,
    sipp_status: str | None = None,
    occupation_leaders: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write the variance addon markdown summary."""
    available_years_text = (
        ", ".join(str(year) for year in available_years) if available_years else "none"
    )
    lines = [
        "# Variance Addon Summary",
        "",
        f"- ACS years available in this run: {available_years_text}",
        f"- Inventory rows tracked: {len(inventory_usage)}",
        f"- Missing required inputs: {len(missing_inputs)}",
        "",
        "## Reused from repro baseline",
    ]
    if reused_outputs:
        lines.extend([f"- `{_relpath(path)}`" for path in reused_outputs])
    else:
        lines.append("- None")

    lines.extend(["", "## Newly added for the variance addon"])
    if addon_outputs:
        lines.extend([f"- `{_relpath(path)}`" for path in addon_outputs])
    else:
        lines.append("- None")

    lines.extend(["", "## Missing inputs / skips"])
    if missing_inputs:
        lines.extend([f"- {item}" for item in missing_inputs])
    else:
        lines.append("- None")

    lines.extend(["", "## Mechanism / robustness status"])
    lines.append(f"- ATUS: {atus_status or 'not run'}")
    lines.append(f"- SIPP: {sipp_status or 'not run'}")

    if onet_coverage is not None and not onet_coverage.empty:
        lines.extend(["", "## O*NET coverage"])
        for row in onet_coverage.itertuples(index=False):
            lines.append(
                f"- {row.survey_year}: {row.n_matched}/{row.n_obs} matched "
                f"({row.match_rate:.1%}) via `{row.merge_key}`"
            )

    if optional_validation is not None and not optional_validation.empty:
        lines.extend(["", "## Optional validation status"])
        for row in optional_validation.itertuples(index=False):
            lines.append(f"- {row.dataset}: {row.status}: {row.note}")

    if occupation_leaders is not None and not occupation_leaders.empty:
        lines.extend(["", "## Occupation-level variability leaders"])
        top_rows = (
            occupation_leaders.sort_values(["outcome", "leaderboard", "rank"])
            .groupby(["outcome", "leaderboard"], observed=True, sort=False)
            .head(3)
        )
        for row in top_rows.itertuples(index=False):
            outcome = _safe_value(row, "outcome", "")
            leaderboard = _safe_value(
                row,
                "leaderboard",
                _safe_value(row, "rank_group", ""),
            )
            rank = _safe_value(row, "rank", "")
            occupation_code = _safe_value(
                row,
                "occupation_harmonized_code",
                _safe_value(row, "occupation_code", ""),
            )
            occupation_title = _safe_value(
                row,
                "occupation_harmonized_title",
                _safe_value(row, "occupation_title", ""),
            )
            n_obs = _safe_value(row, "n_obs", "")
            if occupation_title:
                occupation_label = f"`{occupation_code}` ({occupation_title})"
            else:
                occupation_label = f"`{occupation_code}`"
            if hasattr(row, "raw_variance_ratio"):
                raw_ratio = _safe_value(row, "raw_variance_ratio", "")
                residual_ratio = _safe_value(row, "residual_variance_ratio", "")
                top10_gap = _safe_value(row, "top10_share_gap_pp", "")
                lines.append(
                    f"- {outcome} / {leaderboard} #{rank}: occupation {occupation_label} "
                    f"(raw_ratio={raw_ratio}, residual_ratio={residual_ratio}, "
                    f"top10_gap_pp={top10_gap}, n={n_obs})"
                )
            else:
                metric = _safe_value(row, "metric", "")
                value = _safe_value(row, "value", "")
                lines.append(
                    f"- {outcome} / {leaderboard} #{rank}: occupation {occupation_label} "
                    f"({metric}={value}, n={n_obs})"
                )

    if notes:
        lines.extend(["", "## Notes"])
        lines.extend([f"- {note}" for note in notes])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_variance_release_manifest(
    output_paths: list[Path],
    inventory_usage: pd.DataFrame | None = None,
    optional_validation: pd.DataFrame | None = None,
    reused_outputs: list[Path] | None = None,
) -> dict:
    """Create a machine-readable manifest for variance-addon outputs."""
    manifest = {
        "project": "paygap_variance_addon",
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "outputs": [_artifact_metadata(path) for path in output_paths if path.exists()],
        "inventory_inputs": [],
        "optional_validation_status": [],
        "reused_outputs": [_relpath(path) for path in (reused_outputs or []) if path.exists()],
    }
    if inventory_usage is not None and not inventory_usage.empty:
        manifest["inventory_inputs"] = inventory_usage.fillna("").to_dict(orient="records")
    if optional_validation is not None and not optional_validation.empty:
        manifest["optional_validation_status"] = optional_validation.fillna("").to_dict(
            orient="records"
        )
    return manifest


def write_variance_release_manifest(manifest: dict, output_path: Path) -> Path:
    """Write the variance release manifest to JSON and markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Variance Release Manifest",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Platform: `{manifest['platform']}`",
        f"- Output artifacts tracked: {len(manifest['outputs'])}",
        f"- Reused repro artifacts: {len(manifest.get('reused_outputs', []))}",
        "",
        "## Output artifacts",
    ]
    for item in manifest["outputs"]:
        lines.append(
            f"- `{item['path']}`: {item['size_bytes']} bytes, sha256 `{item['sha256'][:12]}`, "
            f"rows={item.get('row_count', 'n/a')}"
        )
    if manifest.get("reused_outputs"):
        lines.extend(["", "## Reused outputs"])
        lines.extend([f"- `{path}`" for path in manifest["reused_outputs"]])
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


def validate_variance_output_schemas(schema_path: Path) -> dict:
    """Compare current variance-addon output columns against a schema snapshot."""
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


def write_variance_schema_check(report: dict, output_path: Path) -> Path:
    """Write variance schema validation results to JSON and markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Variance Schema Check",
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


def _artifact_metadata(path: Path) -> dict:
    item = {
        "path": _relpath(path),
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


def _relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def _safe_value(row, field: str, default):
    value = getattr(row, field, default)
    if pd.isna(value):
        return default
    return value
