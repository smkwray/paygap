"""Dedicated backend runner for the variance addon."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gender_gap.crosswalks.occupation_crosswalks import attach_occupation_metadata
from gender_gap.features.occupation_context import (
    build_onet_indices,
    build_onet_merge_coverage,
    merge_onet_context,
)
from gender_gap.features.reproductive import (
    add_fertility_risk_features,
    add_repro_interactions,
    add_reproductive_features,
)
from gender_gap.models.atus_mechanisms import build_atus_mechanism_table
from gender_gap.models.fertility_risk import build_same_sex_placebos, run_fertility_risk_penalty
from gender_gap.models.ols import (
    BLOCK_DEFINITIONS,
    _fit_weighted_least_squares,
    coefficient_table,
    results_to_dataframe,
    run_sequential_ols,
)
from gender_gap.models.selection import (
    SELECTION_BLOCKS,
    _build_controls_matrix,
    _fit_weighted_binomial,
)
from gender_gap.models.sipp_robustness import build_sipp_robustness_table
from gender_gap.models.variance_suite import run_variance_suite
from gender_gap.reporting.repro import (
    build_local_inventory_report,
    build_optional_validation_status,
    build_repro_inventory_usage,
    write_local_inventory_report,
    write_nlsy_validation_output,
)
from gender_gap.reporting.variance import (
    build_variance_release_manifest,
    validate_variance_output_schemas,
    write_variance_inventory_report,
    write_variance_release_manifest,
    write_variance_schema_check,
    write_variance_summary,
)
from gender_gap.repro import (
    _compress_panel,
    _final_panel_columns,
    _resolve_onet_dir,
)
from gender_gap.settings import PROJECT_ROOT, load_variance_config, shared_source_path

VARIANCE_TABLE_COLUMNS = ["suite", "outcome", "stratifier", "stratum", "metric", "value", "n_obs"]
OCCUPATION_DISPERSION_COLUMNS = [
    *VARIANCE_TABLE_COLUMNS,
    "occupation_harmonized_code",
    "occupation_harmonized_title",
    "occupation_harmonization_type",
    "occupation_title_vintage",
    "soc_major_group",
    "soc_major_label",
]
OCCUPATION_LEADERS_COLUMNS = [
    "leaderboard",
    "rank",
    "outcome",
    "occupation_harmonized_code",
    "occupation_harmonized_title",
    "occupation_harmonization_type",
    "occupation_title_vintage",
    "soc_major_group",
    "soc_major_label",
    "n_obs",
    "raw_variance_ratio",
    "residual_variance_ratio",
    "female_p90_p10",
    "male_p90_p10",
    "female_top10_share",
    "male_top10_share",
    "female_top5_share",
    "male_top5_share",
    "raw_variance_gap_from_parity",
    "residual_variance_gap_from_parity",
    "top10_share_gap_pp",
    "top5_share_gap_pp",
]
SELECTION_TABLE_COLUMNS = [
    "status",
    "reason",
    "suite",
    "outcome",
    "stratifier",
    "stratum",
    "metric",
    "value",
    "n_obs",
]
SELECTION_LPM_ROW_THRESHOLD = 1_000_000
SELECTION_PANEL_COLUMNS = [
    "female",
    "age",
    "age_sq",
    "race_ethnicity",
    "education_level",
    "marital_status",
    "number_children",
    "children_under_5",
    "state_fips",
    "person_weight",
    "employment_indicator",
    "employed",
    "hourly_wage_real",
    "log_hourly_wage_real",
    "survey_year",
]
ONET_COVERAGE_COLUMNS = ["survey_year", "n_obs", "n_matched", "match_rate", "merge_key"]
DEFAULT_OCCUPATION_MIN_N = 5000
DEFAULT_OCCUPATION_TOP_K = 25

REPRO_BASELINE_OUTPUTS = {
    "gap_ladder": "acs_gap_ladder_extended.csv",
    "gap_ladder_by_year": "acs_gap_ladder_by_year.csv",
    "fertility_penalty": "acs_fertility_risk_penalty.csv",
    "fertility_quartiles": "acs_fertility_risk_by_quartile.csv",
    "same_sex_placebos": "acs_same_sex_placebos.csv",
    "variance_suite": "acs_variance_suite.csv",
    "tail_metrics": "acs_tail_metrics.csv",
    "atus_mechanisms": "atus_mechanisms.csv",
    "sipp_robustness": "sipp_robustness.csv",
}


def run_variance_addon(config_path: Path | None = None) -> dict[str, Path]:
    """Run the dedicated variance addon using cached repro assets when available."""
    cfg = load_variance_config(config_path)
    paths = cfg.get("paths", {})
    analysis = cfg.get("analysis", {})
    acs_years = cfg.get("datasets", {}).get("acs", {}).get("years", [])
    required_onet = cfg.get("datasets", {}).get("onet", {}).get("required_files", [])

    results_dir = PROJECT_ROOT / paths.get("results_dir", "results/variance")
    report_path = PROJECT_ROOT / paths.get("report_path", "reports/variance_addon_summary.md")
    inventory_output = PROJECT_ROOT / paths.get(
        "inventory_usage_output", "results/diagnostics/variance_inventory_usage.csv"
    )
    inventory_report_output = PROJECT_ROOT / paths.get(
        "inventory_usage_report", "reports/variance_inventory_usage.md"
    )
    optional_validation_output = PROJECT_ROOT / paths.get(
        "optional_validation_output", "results/diagnostics/variance_optional_validation_status.csv"
    )
    inventory_config = PROJECT_ROOT / paths.get(
        "inventory_config", "inventory/inventory_paths.yaml"
    )
    local_inventory_output = PROJECT_ROOT / paths.get(
        "local_inventory_output", "diagnostics/variance_local_inventory_report.json"
    )
    atus_report_output = PROJECT_ROOT / paths.get(
        "atus_report_output", "reports/atus_variance_mechanisms.md"
    )
    onet_coverage_output = PROJECT_ROOT / paths.get(
        "onet_merge_coverage", "results/diagnostics/variance_onet_merge_coverage.csv"
    )
    release_manifest_output = PROJECT_ROOT / paths.get(
        "release_manifest_output", "diagnostics/variance_release_manifest.json"
    )
    schema_snapshot_path = PROJECT_ROOT / paths.get(
        "schema_snapshot_path", "configs/variance_output_schemas.json"
    )
    schema_check_output = PROJECT_ROOT / paths.get(
        "schema_check_output", "diagnostics/variance_schema_check.json"
    )
    occupation_harmonization_output = PROJECT_ROOT / paths.get(
        "occupation_harmonization_output",
        "results/diagnostics/variance_occupation_harmonization_map.csv",
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    inventory_output.parent.mkdir(parents=True, exist_ok=True)
    local_inventory_output.parent.mkdir(parents=True, exist_ok=True)
    onet_coverage_output.parent.mkdir(parents=True, exist_ok=True)
    occupation_harmonization_output.parent.mkdir(parents=True, exist_ok=True)

    inventory_usage = _build_variance_inventory_usage(acs_years, required_onet)
    inventory_usage.to_csv(inventory_output, index=False)
    write_variance_inventory_report(inventory_usage, inventory_report_output)

    optional_validation = build_optional_validation_status()
    optional_validation.to_csv(optional_validation_output, index=False)

    outputs: dict[str, Path] = {
        "inventory_usage": inventory_output,
        "inventory_report": inventory_report_output,
        "optional_validation_status": optional_validation_output,
    }
    reused_outputs: list[Path] = []
    addon_outputs: list[Path] = []
    notes: list[str] = []

    if inventory_config.exists():
        local_inventory_report = build_local_inventory_report(inventory_config)
        write_local_inventory_report(local_inventory_report, local_inventory_output)
        outputs["local_inventory_report_json"] = local_inventory_output
        outputs["local_inventory_report_md"] = local_inventory_output.with_suffix(".md")
        addon_outputs.extend([local_inventory_output, local_inventory_output.with_suffix(".md")])

    panel = _load_cached_repro_panel()
    available_years: list[int] = []
    onet_coverage = pd.DataFrame(columns=ONET_COVERAGE_COLUMNS)

    if panel is not None:
        panel, onet_coverage = _prepare_panel(panel, required_onet)
        available_years = sorted(
            int(year)
            for year in pd.to_numeric(panel["survey_year"], errors="coerce").dropna().unique()
        )
        _write_panel_derived_outputs(
            panel=panel,
            results_dir=results_dir,
            analysis=analysis,
            outputs=outputs,
            addon_outputs=addon_outputs,
            occupation_harmonization_output=occupation_harmonization_output,
        )
        if not onet_coverage.empty:
            onet_coverage.to_csv(onet_coverage_output, index=False)
            outputs["onet_merge_coverage"] = onet_coverage_output
            addon_outputs.append(onet_coverage_output)
    else:
        available_years = _promote_repro_baseline(results_dir, outputs, reused_outputs)
        if "variance_suite" not in outputs or "gap_ladder" not in outputs:
            raise FileNotFoundError(
                "Variance addon requires either "
                "`data/processed/acs_repro_panel.parquet` or the existing repro "
                "baseline outputs under `results/repro/`. Build the cached panel "
                "on the remote host before rerunning."
            )
        notes.append(
            "No cached `data/processed/acs_repro_panel.parquet` was available "
            "locally, so the addon promoted existing repro outputs instead of "
            "rebuilding the pooled ACS panel."
        )
        notes.append(
            "Full multi-outcome variance reruns remain supported once a cached "
            "repro panel is materialized, which is the recommended path on the "
            "remote host for large ACS builds."
        )
        _write_split_variance_views_from_repro(
            results_dir,
            outputs,
            addon_outputs,
            occupation_harmonization_output=occupation_harmonization_output,
        )
        onet_coverage.to_csv(onet_coverage_output, index=False)
        outputs["onet_merge_coverage"] = onet_coverage_output
        addon_outputs.append(onet_coverage_output)

    selection_panel = _load_selection_panel()
    if selection_panel is None:
        selection_panel = panel
    selection_path = _write_selection_corrected_variance(selection_panel, results_dir)
    outputs["selection_corrected_variance"] = selection_path
    addon_outputs.append(selection_path)

    atus_status, atus_path = _write_atus_outputs(results_dir, atus_report_output)
    outputs["atus_mechanisms"] = atus_path
    outputs["atus_report"] = atus_report_output
    addon_outputs.extend([atus_path, atus_report_output])

    sipp_status, sipp_path = _write_sipp_outputs(results_dir)
    outputs["sipp_robustness"] = sipp_path
    addon_outputs.append(sipp_path)

    _write_question_pack_outputs(
        results_dir=results_dir, outputs=outputs, addon_outputs=addon_outputs
    )

    nlsy_validation_path = write_nlsy_validation_output(results_dir / "nlsy_validation.csv")
    if nlsy_validation_path is not None:
        outputs["nlsy_validation"] = nlsy_validation_path
        reused_outputs.append(nlsy_validation_path)

    missing_inputs = inventory_usage.loc[
        inventory_usage["status"] == "missing", "asset_name"
    ].tolist()
    write_variance_summary(
        output_path=report_path,
        available_years=available_years,
        inventory_usage=inventory_usage,
        reused_outputs=reused_outputs,
        addon_outputs=addon_outputs,
        missing_inputs=missing_inputs,
        optional_validation=optional_validation,
        onet_coverage=onet_coverage,
        atus_status=atus_status,
        sipp_status=sipp_status,
        occupation_leaders=(
            pd.read_csv(outputs["occupation_variability_leaders"])
            if "occupation_variability_leaders" in outputs
            and outputs["occupation_variability_leaders"].exists()
            else None
        ),
        notes=notes,
    )
    outputs["summary_report"] = report_path

    manifest = build_variance_release_manifest(
        output_paths=list(outputs.values()),
        inventory_usage=inventory_usage,
        optional_validation=optional_validation,
        reused_outputs=reused_outputs,
    )
    write_variance_release_manifest(manifest, release_manifest_output)
    outputs["release_manifest_json"] = release_manifest_output
    outputs["release_manifest_md"] = release_manifest_output.with_suffix(".md")

    if schema_snapshot_path.exists():
        schema_report = validate_variance_output_schemas(schema_snapshot_path)
        write_variance_schema_check(schema_report, schema_check_output)
        outputs["schema_check_json"] = schema_check_output
        outputs["schema_check_md"] = schema_check_output.with_suffix(".md")

    return outputs


def _build_variance_inventory_usage(
    required_acs_years: list[int], onet_required_files: list[str]
) -> pd.DataFrame:
    usage = build_repro_inventory_usage(required_acs_years, onet_required_files).copy()
    extra_rows = []
    atus_candidates = [
        (
            "atus",
            "atus_analysis_ready.parquet",
            PROJECT_ROOT / "data" / "processed" / "atus_analysis_ready.parquet",
            shared_source_path(
                "bls",
                "atus",
                "2003_2024",
                "paygap",
                "processed",
                "atus",
                "atus_analysis_ready.parquet",
            ),
        ),
        (
            "atus",
            "atus_respondent.parquet",
            PROJECT_ROOT / "data" / "raw" / "atus" / "atus_respondent.parquet",
            shared_source_path(
                "bls", "atus", "2003_2024", "paygap", "raw", "atus", "atus_respondent.parquet"
            ),
        ),
        (
            "atus",
            "atus_roster.parquet",
            PROJECT_ROOT / "data" / "raw" / "atus" / "atus_roster.parquet",
            shared_source_path(
                "bls", "atus", "2003_2024", "paygap", "raw", "atus", "atus_roster.parquet"
            ),
        ),
        (
            "sipp",
            "sipp_standardized.parquet",
            PROJECT_ROOT / "data" / "processed" / "sipp_standardized.parquet",
            PROJECT_ROOT / "data" / "processed" / "sipp_standardized.parquet",
        ),
        (
            "sipp",
            "pu2024.dta",
            PROJECT_ROOT / "data" / "raw" / "sipp" / "pu2024.dta",
            shared_source_path(
                "misc", "large_payloads", "wave3c", "paygap", "raw", "sipp", "pu2024.dta"
            ),
        ),
        (
            "occupation_crosswalk",
            "2018-occupation-code-list-and-crosswalk.xlsx",
            PROJECT_ROOT
            / "data"
            / "raw"
            / "crosswalks"
            / "2018-occupation-code-list-and-crosswalk.xlsx",
            shared_source_path(
                "census",
                "industry_occupation",
                "2018",
                "2018-occupation-code-list-and-crosswalk.xlsx",
            ),
        ),
        (
            "occupation_crosswalk",
            "census_occupation_code_lookup_2010_2018.csv",
            PROJECT_ROOT
            / "data"
            / "processed"
            / "crosswalks"
            / "census_occupation_code_lookup_2010_2018.csv",
            shared_source_path(
                "census",
                "industry_occupation",
                "2018",
                "census_occupation_code_lookup_2010_2018.csv",
            ),
        ),
    ]
    for asset_group, asset_name, local_path, shared_path in atus_candidates:
        chosen = shared_path if shared_path.exists() else local_path
        extra_rows.append(
            {
                "asset_group": asset_group,
                "asset_name": asset_name,
                "status": "present" if chosen.exists() else "missing",
                "legacy_path": str(local_path.relative_to(PROJECT_ROOT))
                if local_path.is_relative_to(PROJECT_ROOT)
                else str(local_path),
                "canonical_path": str(shared_path.relative_to(shared_path.parents[4]))
                if shared_path.exists()
                else "",
                "note": "variance_addon",
            }
        )
    return pd.concat([usage, pd.DataFrame(extra_rows)], ignore_index=True)


def _load_cached_repro_panel() -> pd.DataFrame | None:
    panel_path = PROJECT_ROOT / "data" / "processed" / "acs_repro_panel.parquet"
    if not panel_path.exists():
        return None
    return pd.read_parquet(panel_path)


def _load_selection_panel() -> pd.DataFrame | None:
    panel_path = PROJECT_ROOT / "data" / "processed" / "acs_repro_panel.parquet"
    if not panel_path.exists():
        return None
    available = set(pq.ParquetFile(panel_path).schema_arrow.names)
    columns = [column for column in SELECTION_PANEL_COLUMNS if column in available]
    if not columns:
        return None
    panel = pd.read_parquet(panel_path, columns=columns)
    for column in ["race_ethnicity", "education_level", "marital_status"]:
        if column in panel.columns:
            panel[column] = panel[column].astype("category")
    return panel


def _prepare_panel(
    panel: pd.DataFrame, required_onet: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = add_reproductive_features(panel)
    onet_coverage = (
        build_onet_merge_coverage(enriched)
        if "job_rigidity" in enriched.columns
        else pd.DataFrame(columns=ONET_COVERAGE_COLUMNS)
    )
    if "job_rigidity" not in enriched.columns:
        onet_dir = _resolve_onet_dir(required_onet)
        onet_indices = build_onet_indices(
            onet_dir=onet_dir, recipe_path=PROJECT_ROOT / "configs" / "onet_index_recipe.yaml"
        )
        enriched, onet_coverage = merge_onet_context(enriched, onet_indices)
    enriched = add_fertility_risk_features(enriched)
    enriched = add_repro_interactions(enriched)
    enriched = _ensure_logged_outcomes(enriched)
    for column in ["autonomy", "time_pressure"]:
        quartile = f"{column}_quartile"
        if column in enriched.columns and quartile not in enriched.columns:
            enriched[quartile] = _quartiles(enriched[column])
    keep_columns = _final_panel_columns().union({"autonomy_quartile", "time_pressure_quartile"})
    return _compress_panel(enriched, keep_columns), onet_coverage


def _write_panel_derived_outputs(
    panel: pd.DataFrame,
    results_dir: Path,
    analysis: dict,
    outputs: dict[str, Path],
    addon_outputs: list[Path],
    occupation_harmonization_output: Path,
) -> None:
    blocks = {
        name: controls for name, controls in BLOCK_DEFINITIONS.items() if name.startswith("M")
    }

    ols = run_sequential_ols(panel, weight_col="person_weight", blocks=blocks)
    ladder_path = results_dir / "acs_gap_ladder_extended.csv"
    results_to_dataframe(ols).to_csv(ladder_path, index=False)
    outputs["gap_ladder"] = ladder_path
    addon_outputs.append(ladder_path)

    by_year_rows = []
    for year, year_df in panel.groupby("survey_year", observed=True):
        year_ols = run_sequential_ols(year_df, weight_col="person_weight", blocks=blocks)
        year_result = results_to_dataframe(year_ols)
        year_result.insert(0, "year", year)
        by_year_rows.append(year_result)
    by_year_path = results_dir / "acs_gap_ladder_by_year.csv"
    pd.concat(by_year_rows, ignore_index=True).to_csv(by_year_path, index=False)
    outputs["gap_ladder_by_year"] = by_year_path
    addon_outputs.append(by_year_path)

    interaction_path = results_dir / "acs_onet_interactions.csv"
    coefficient_table(
        panel,
        model_name="M8_reproductive_x_job_context",
        weight_col="person_weight",
    ).loc[lambda d: d["term"].str.contains("female_x_", na=False)].to_csv(
        interaction_path, index=False
    )
    outputs["onet_interactions"] = interaction_path
    addon_outputs.append(interaction_path)

    penalty_df, quartile_df = run_fertility_risk_penalty(panel)
    penalty_path = results_dir / "acs_fertility_risk_penalty.csv"
    quartile_path = results_dir / "acs_fertility_risk_by_quartile.csv"
    placebos_path = results_dir / "acs_same_sex_placebos.csv"
    penalty_df.to_csv(penalty_path, index=False)
    quartile_df.to_csv(quartile_path, index=False)
    build_same_sex_placebos(panel).to_csv(placebos_path, index=False)
    outputs["fertility_penalty"] = penalty_path
    outputs["fertility_quartiles"] = quartile_path
    outputs["same_sex_placebos"] = placebos_path
    addon_outputs.extend([penalty_path, quartile_path, placebos_path])

    outcome_frames = []
    for outcome in analysis.get("outcomes", ["log_hourly_wage_real"]):
        if outcome not in panel.columns:
            continue
        frame = run_variance_suite(
            panel,
            outcome=outcome,
            weight_col="person_weight",
            stratifiers=[
                "survey_year",
                *analysis.get("reproductive_stratifiers", []),
                *analysis.get("onet_stratifiers", []),
            ],
        ).copy()
        if frame.empty:
            continue
        frame.insert(0, "outcome", outcome)
        frame.insert(0, "suite", frame["stratifier"].map(_suite_for_stratifier))
        outcome_frames.append(frame)

    variance_path = results_dir / "acs_variance_suite.csv"
    tail_path = results_dir / "acs_tail_metrics.csv"
    reproductive_path = results_dir / "acs_reproductive_dispersion.csv"
    onet_path = results_dir / "acs_onet_dispersion.csv"
    occupation_path = results_dir / "acs_occupation_dispersion.csv"
    occupation_leaders_path = results_dir / "acs_occupation_variability_leaders.csv"
    variance_df = (
        pd.concat(outcome_frames, ignore_index=True)
        if outcome_frames
        else pd.DataFrame(columns=VARIANCE_TABLE_COLUMNS)
    )
    variance_df.to_csv(variance_path, index=False)
    variance_df.loc[
        variance_df["metric"].str.contains("top|p90|p95", case=False, regex=True, na=False)
    ].to_csv(tail_path, index=False)
    variance_df.loc[
        variance_df["stratifier"].isin(analysis.get("reproductive_stratifiers", []))
    ].to_csv(reproductive_path, index=False)
    variance_df.loc[variance_df["stratifier"].isin(analysis.get("onet_stratifiers", []))].to_csv(
        onet_path, index=False
    )
    occupation_panel = _prepare_harmonized_occupation_panel(panel, analysis)
    occupation_df = _build_occupation_dispersion(occupation_panel, analysis)
    occupation_df.to_csv(occupation_path, index=False)
    _build_occupation_leaders(occupation_df, analysis).to_csv(occupation_leaders_path, index=False)
    _build_occupation_harmonization_map(occupation_panel).to_csv(
        occupation_harmonization_output, index=False
    )
    outputs["variance_suite"] = variance_path
    outputs["tail_metrics"] = tail_path
    outputs["reproductive_dispersion"] = reproductive_path
    outputs["onet_dispersion"] = onet_path
    outputs["occupation_dispersion"] = occupation_path
    outputs["occupation_variability_leaders"] = occupation_leaders_path
    outputs["occupation_harmonization_map"] = occupation_harmonization_output
    addon_outputs.extend(
        [
            variance_path,
            tail_path,
            reproductive_path,
            onet_path,
            occupation_path,
            occupation_leaders_path,
            occupation_harmonization_output,
        ]
    )


def _promote_repro_baseline(
    results_dir: Path, outputs: dict[str, Path], reused_outputs: list[Path]
) -> list[int]:
    available_years: list[int] = []
    repro_dir = PROJECT_ROOT / "results" / "repro"
    for key, filename in REPRO_BASELINE_OUTPUTS.items():
        src = repro_dir / filename
        if not src.exists():
            continue
        dest = results_dir / filename
        shutil.copy2(src, dest)
        outputs[key] = dest
        reused_outputs.append(src)
        if filename == "acs_gap_ladder_by_year.csv":
            year_df = pd.read_csv(dest)
            if "year" in year_df.columns:
                available_years = sorted(
                    int(year)
                    for year in pd.to_numeric(year_df["year"], errors="coerce").dropna().unique()
                )
    return available_years


def _write_split_variance_views_from_repro(
    results_dir: Path,
    outputs: dict[str, Path],
    addon_outputs: list[Path],
    occupation_harmonization_output: Path,
) -> None:
    variance_path = results_dir / "acs_variance_suite.csv"
    if not variance_path.exists():
        variance_df = pd.DataFrame(columns=VARIANCE_TABLE_COLUMNS)
    else:
        variance_df = pd.read_csv(variance_path)
        if "outcome" not in variance_df.columns:
            variance_df.insert(0, "outcome", "log_hourly_wage_real")
        if "suite" not in variance_df.columns:
            variance_df.insert(0, "suite", variance_df["stratifier"].map(_suite_for_stratifier))
        variance_df.to_csv(variance_path, index=False)

    tail_path = results_dir / "acs_tail_metrics.csv"
    if tail_path.exists():
        tail_df = pd.read_csv(tail_path)
        if "outcome" not in tail_df.columns:
            tail_df.insert(0, "outcome", "log_hourly_wage_real")
        if "suite" not in tail_df.columns:
            tail_df.insert(0, "suite", tail_df["stratifier"].map(_suite_for_stratifier))
        tail_df.to_csv(tail_path, index=False)

    reproductive_path = results_dir / "acs_reproductive_dispersion.csv"
    onet_path = results_dir / "acs_onet_dispersion.csv"
    occupation_path = results_dir / "acs_occupation_dispersion.csv"
    occupation_leaders_path = results_dir / "acs_occupation_variability_leaders.csv"
    variance_df.loc[
        variance_df["stratifier"].isin(
            ["reproductive_stage", "fertility_risk_quartile", "couple_type"]
        )
    ].to_csv(reproductive_path, index=False)
    variance_df.loc[
        variance_df["stratifier"].isin(
            ["job_rigidity_quartile", "autonomy_quartile", "time_pressure_quartile"]
        )
    ].to_csv(onet_path, index=False)
    occupation_rows = variance_df.loc[variance_df["stratifier"].isin(["occupation_code"])].copy()
    if occupation_rows.empty:
        occupation_df = pd.DataFrame(columns=OCCUPATION_DISPERSION_COLUMNS)
        occupation_map = pd.DataFrame(columns=_occupation_harmonization_columns())
    else:
        occupation_rows = _attach_occupation_metadata_year_aware(
            occupation_rows,
            code_col="stratum",
            survey_year_col="survey_year",
            require_year_context=False,
        )
        occupation_map = _build_occupation_harmonization_map(occupation_rows)
        occupation_df = _harmonize_occupation_surface(occupation_rows, strict=False)
    occupation_df.to_csv(occupation_path, index=False)
    occupation_map.to_csv(occupation_harmonization_output, index=False)
    _build_occupation_leaders(pd.read_csv(occupation_path), {}).to_csv(
        occupation_leaders_path, index=False
    )
    outputs["reproductive_dispersion"] = reproductive_path
    outputs["onet_dispersion"] = onet_path
    outputs["occupation_dispersion"] = occupation_path
    outputs["occupation_variability_leaders"] = occupation_leaders_path
    outputs["occupation_harmonization_map"] = occupation_harmonization_output
    addon_outputs.extend(
        [
            reproductive_path,
            onet_path,
            occupation_path,
            occupation_leaders_path,
            occupation_harmonization_output,
        ]
    )


def _write_question_pack_outputs(
    results_dir: Path, outputs: dict[str, Path], addon_outputs: list[Path]
) -> None:
    tail_path = results_dir / "acs_tail_contrast_summary.csv"
    regime_path = results_dir / "acs_year_regime_variance_summary.csv"
    soc_path = results_dir / "acs_soc_group_leaderboard_counts.csv"
    fertility_path = results_dir / "acs_fertility_risk_variance_bridge.csv"

    _build_tail_contrast_summary(results_dir).to_csv(tail_path, index=False)
    _build_year_regime_variance_summary(results_dir).to_csv(regime_path, index=False)
    _build_soc_group_leaderboard_counts(results_dir).to_csv(soc_path, index=False)
    _build_fertility_risk_variance_bridge(results_dir).to_csv(fertility_path, index=False)

    outputs["tail_contrast_summary"] = tail_path
    outputs["year_regime_variance_summary"] = regime_path
    outputs["soc_group_leaderboard_counts"] = soc_path
    outputs["fertility_risk_variance_bridge"] = fertility_path
    addon_outputs.extend([tail_path, regime_path, soc_path, fertility_path])


def _write_selection_corrected_variance(panel: pd.DataFrame | None, results_dir: Path) -> Path:
    path = results_dir / "acs_selection_corrected_variance.csv"
    if panel is None:
        pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": (
                        "selection-corrected variance requires a cached repro "
                        "panel; rerun on the remote host once "
                        "`data/processed/acs_repro_panel.parquet` is "
                        "materialized."
                    ),
                    "suite": "V2_selection_corrected",
                    "outcome": "log_hourly_wage_real",
                    "stratifier": "overall",
                    "stratum": "all",
                    "metric": "selection_corrected_variance",
                    "value": np.nan,
                    "n_obs": 0,
                }
            ],
            columns=SELECTION_TABLE_COLUMNS,
        ).to_csv(path, index=False)
        return path

    employed_col = (
        "employment_indicator" if "employment_indicator" in panel.columns else "employed"
    )
    selection_columns = [col for col in SELECTION_PANEL_COLUMNS if col in panel.columns]
    model_df = panel.loc[:, selection_columns].copy()
    for column in ["race_ethnicity", "education_level", "marital_status"]:
        if column in model_df.columns:
            model_df[column] = model_df[column].astype("category")
    if "log_hourly_wage_real" not in model_df.columns and "hourly_wage_real" in model_df.columns:
        hourly = pd.to_numeric(model_df["hourly_wage_real"], errors="coerce")
        model_df["log_hourly_wage_real"] = np.log(hourly.where(hourly > 0))
    if (
        employed_col not in model_df.columns
        or "person_weight" not in model_df.columns
        or "log_hourly_wage_real" not in model_df.columns
    ):
        pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": (
                        "panel is missing employment or wage fields needed for "
                        "IPW variance adjustment."
                    ),
                    "suite": "V2_selection_corrected",
                    "outcome": "log_hourly_wage_real",
                    "stratifier": "overall",
                    "stratum": "all",
                    "metric": "selection_corrected_variance",
                    "value": np.nan,
                    "n_obs": 0,
                }
            ],
            columns=SELECTION_TABLE_COLUMNS,
        ).to_csv(path, index=False)
        return path
    weight_mask = pd.to_numeric(model_df["person_weight"], errors="coerce").gt(0)
    employed_mask = pd.to_numeric(model_df[employed_col], errors="coerce").notna()
    model_df = model_df.loc[weight_mask & employed_mask].copy()
    if model_df.empty:
        return _write_selection_corrected_variance(None, results_dir)

    X = _build_controls_matrix(model_df, SELECTION_BLOCKS["S2"])
    model_df = model_df.loc[X.index].copy()
    weights = pd.to_numeric(model_df["person_weight"], errors="coerce").astype(float)
    y_emp = pd.to_numeric(model_df[employed_col], errors="coerce").astype(float)
    fit = _fit_selection_model(y_emp, X, weights)
    p_actual = pd.Series(np.clip(fit.predict(X), 0.02, 1.0), index=X.index, dtype=float)

    worker_mask = (
        pd.to_numeric(model_df[employed_col], errors="coerce").eq(1)
        & pd.to_numeric(model_df["log_hourly_wage_real"], errors="coerce").notna()
    )
    worker_df = model_df.loc[worker_mask].copy()
    if worker_df.empty:
        return _write_selection_corrected_variance(None, results_dir)

    worker_df["selection_ipw"] = weights.loc[worker_df.index] / p_actual.loc[worker_df.index]
    variance_df = run_variance_suite(
        worker_df,
        outcome="log_hourly_wage_real",
        weight_col="selection_ipw",
        stratifiers=["survey_year"],
    ).copy()
    if variance_df.empty:
        return _write_selection_corrected_variance(None, results_dir)

    variance_df.insert(0, "outcome", "log_hourly_wage_real")
    variance_df.insert(0, "suite", "V2_selection_corrected")
    variance_df.insert(0, "reason", "")
    variance_df.insert(0, "status", "ok")
    variance_df = variance_df.loc[:, SELECTION_TABLE_COLUMNS]
    variance_df.to_csv(path, index=False)
    return path


def _ensure_logged_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log_hourly_wage_real" not in out.columns and "hourly_wage_real" in out.columns:
        hourly = pd.to_numeric(out["hourly_wage_real"], errors="coerce")
        out["log_hourly_wage_real"] = np.log(hourly.where(hourly > 0))
    if "log_annual_earnings_real" not in out.columns and "annual_earnings_real" in out.columns:
        annual = pd.to_numeric(out["annual_earnings_real"], errors="coerce")
        out["log_annual_earnings_real"] = np.log(annual.where(annual > 0))
    return out


def _fit_selection_model(y_emp: pd.Series, X: pd.DataFrame, weights: pd.Series):
    """Use a lower-memory weighted LPM for very large panels."""
    if len(X) < SELECTION_LPM_ROW_THRESHOLD:
        return _fit_weighted_binomial(y_emp, X, weights)

    fit = _fit_weighted_least_squares(y_emp, X, weights)
    params = fit["params"].to_numpy(dtype=float)

    class _LPMResult:
        @staticmethod
        def predict(X_new):
            arr = X_new.to_numpy(dtype=float, copy=False) @ params
            return np.clip(arr, 0.02, 1.0)

    return _LPMResult()


def _write_atus_outputs(results_dir: Path, report_path: Path) -> tuple[str, Path]:
    path = results_dir / "atus_mechanisms.csv"
    if path.exists():
        mechanisms = pd.read_csv(path)
        report_lines = ["# ATUS Variance Mechanisms", ""]
        if "metric" in mechanisms.columns and "reproductive_stage" in mechanisms.columns:
            report_lines.extend(
                [
                    f"- Rows: {len(mechanisms)}",
                    f"- Stages: {mechanisms['reproductive_stage'].nunique(dropna=True)}",
                    f"- Metrics: {mechanisms['metric'].nunique(dropna=True)}",
                ]
            )
        else:
            report_lines.append("- Reused existing ATUS mechanisms output.")
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        return (f"ok: reused existing ATUS mechanisms ({len(mechanisms)} rows)", path)

    processed = _first_existing(
        [
            PROJECT_ROOT / "data" / "processed" / "atus_analysis_ready.parquet",
            shared_source_path(
                "bls",
                "atus",
                "2003_2024",
                "paygap",
                "processed",
                "atus",
                "atus_analysis_ready.parquet",
            ),
            shared_source_path(
                "bls",
                "atus",
                "2003_2023",
                "paygap",
                "processed",
                "atus",
                "atus_analysis_ready.parquet",
            ),
        ]
    )
    respondent = _first_existing(
        [
            PROJECT_ROOT / "data" / "raw" / "atus" / "atus_respondent.parquet",
            shared_source_path(
                "bls", "atus", "2003_2024", "paygap", "raw", "atus", "atus_respondent.parquet"
            ),
            shared_source_path(
                "bls", "atus", "2003_2023", "paygap", "raw", "atus", "atus_respondent.parquet"
            ),
        ]
    )
    roster = _first_existing(
        [
            PROJECT_ROOT / "data" / "raw" / "atus" / "atus_roster.parquet",
            shared_source_path(
                "bls", "atus", "2003_2024", "paygap", "raw", "atus", "atus_roster.parquet"
            ),
            shared_source_path(
                "bls", "atus", "2003_2023", "paygap", "raw", "atus", "atus_roster.parquet"
            ),
        ]
    )
    missing = [
        name
        for name, candidate in {
            "processed": processed,
            "respondent": respondent,
            "roster": roster,
        }.items()
        if candidate is None
    ]
    if missing:
        skipped = pd.DataFrame(
            [{"status": "skipped", "reason": f"ATUS inputs missing: {', '.join(missing)}"}]
        )
        skipped.to_csv(path, index=False)
        report_path.write_text(
            "# ATUS Variance Mechanisms\n\n"
            f"- ATUS mechanism layer was skipped: missing {', '.join(missing)} inputs.\n",
            encoding="utf-8",
        )
        return (f"skipped: missing ATUS inputs ({', '.join(missing)})", path)

    mechanisms = build_atus_mechanism_table(processed, respondent, roster)
    mechanisms.to_csv(path, index=False)
    report_lines = [
        "# ATUS Variance Mechanisms",
        "",
        f"- Rows: {len(mechanisms)}",
        f"- Stages: {mechanisms['reproductive_stage'].nunique(dropna=True)}",
        f"- Metrics: {mechanisms['metric'].nunique(dropna=True)}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return (f"ok: {len(mechanisms)} ATUS mechanism rows", path)


def _write_sipp_outputs(results_dir: Path) -> tuple[str, Path]:
    path = results_dir / "sipp_robustness.csv"
    if path.exists():
        existing = pd.read_csv(path)
        return (f"ok: reused existing SIPP robustness table ({len(existing)} rows)", path)

    standardized = _first_existing(
        [
            PROJECT_ROOT / "data" / "processed" / "sipp_standardized.parquet",
            shared_source_path(
                "misc",
                "large_payloads",
                "wave4",
                "paygap",
                "processed",
                "sipp",
                "sipp_standardized.parquet",
            ),
        ]
    )
    raw = _first_existing(
        [
            PROJECT_ROOT / "data" / "raw" / "sipp" / "pu2024.dta",
            shared_source_path(
                "misc", "large_payloads", "wave3c", "paygap", "raw", "sipp", "pu2024.dta"
            ),
        ]
    )
    missing = [
        name
        for name, candidate in {"standardized": standardized, "raw": raw}.items()
        if candidate is None
    ]
    if missing:
        pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "section": "status",
                    "metric": "missing_inputs",
                    "value": np.nan,
                    "note": f"SIPP inputs missing: {', '.join(missing)}",
                }
            ]
        ).to_csv(path, index=False)
        return (f"skipped: missing SIPP inputs ({', '.join(missing)})", path)

    table = build_sipp_robustness_table(
        standardized_path=standardized, raw_path=raw, survey_year=2023
    )
    table.to_csv(path, index=False)
    return (f"ok: {len(table)} SIPP robustness rows", path)


def _quartiles(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    valid = numeric.notna()
    if valid.sum() < 4:
        return result
    ranked = numeric.loc[valid].rank(method="first")
    result.loc[valid] = pd.qcut(ranked, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype("string")
    return result


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _suite_for_stratifier(stratifier: str) -> str:
    if stratifier in {"overall", "survey_year"}:
        return "V1_raw_residual"
    if stratifier in {"reproductive_stage", "fertility_risk_quartile", "couple_type"}:
        return "V3_reproductive_dispersion"
    if stratifier in {"job_rigidity_quartile", "autonomy_quartile", "time_pressure_quartile"}:
        return "V4_onet_context_dispersion"
    if stratifier in {"occupation_code", "occupation_harmonized_code"}:
        return "V5_occupation_dispersion"
    return "variance_addon"


def _prepare_harmonized_occupation_panel(panel: pd.DataFrame, analysis: dict) -> pd.DataFrame:
    occupation_stratifier = str(analysis.get("occupation_stratifier", "occupation_code"))
    if occupation_stratifier not in panel.columns:
        return pd.DataFrame()
    return _attach_occupation_metadata_year_aware(
        panel,
        code_col=occupation_stratifier,
        survey_year_col="survey_year",
        require_year_context=True,
    )


def _build_occupation_dispersion(occupation_panel: pd.DataFrame, analysis: dict) -> pd.DataFrame:
    if occupation_panel.empty or "occupation_harmonized_code" not in occupation_panel.columns:
        return pd.DataFrame(columns=OCCUPATION_DISPERSION_COLUMNS)

    occupation_min_n = int(analysis.get("occupation_min_n", DEFAULT_OCCUPATION_MIN_N))
    outcome_frames = []
    for outcome in analysis.get("outcomes", ["log_hourly_wage_real"]):
        if outcome not in occupation_panel.columns:
            continue
        frame = run_variance_suite(
            occupation_panel,
            outcome=outcome,
            weight_col="person_weight",
            stratifiers=["occupation_harmonized_code"],
            min_group_n=occupation_min_n,
        ).copy()
        if frame.empty:
            continue
        frame.insert(0, "outcome", outcome)
        frame.insert(0, "suite", frame["stratifier"].map(_suite_for_stratifier))
        outcome_frames.append(frame)

    if not outcome_frames:
        return pd.DataFrame(columns=OCCUPATION_DISPERSION_COLUMNS)
    metadata = _summarize_harmonized_occupation_metadata(occupation_panel)
    dispersion = pd.concat(outcome_frames, ignore_index=True)
    dispersion = dispersion.merge(
        metadata,
        left_on="stratum",
        right_on="occupation_harmonized_code",
        how="left",
    )
    for column in OCCUPATION_DISPERSION_COLUMNS:
        if column not in dispersion.columns:
            dispersion[column] = pd.NA
    return dispersion.loc[:, OCCUPATION_DISPERSION_COLUMNS]


def _build_occupation_leaders(occupation_df: pd.DataFrame, analysis: dict) -> pd.DataFrame:
    if occupation_df.empty:
        return pd.DataFrame(columns=OCCUPATION_LEADERS_COLUMNS)

    top_k = int(analysis.get("occupation_top_k", DEFAULT_OCCUPATION_TOP_K))
    pivot = occupation_df.pivot_table(
        index=[
            "outcome",
            "occupation_harmonized_code",
            "occupation_harmonized_title",
            "occupation_harmonization_type",
            "occupation_title_vintage",
            "soc_major_group",
            "soc_major_label",
            "n_obs",
        ],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    for col in [
        "raw_variance_ratio",
        "residual_variance_ratio",
        "female_p90_p10",
        "male_p90_p10",
        "female_top10_share",
        "male_top10_share",
        "female_top5_share",
        "male_top5_share",
    ]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["raw_variance_gap_from_parity"] = (pivot["raw_variance_ratio"] - 1.0).abs()
    pivot["residual_variance_gap_from_parity"] = (pivot["residual_variance_ratio"] - 1.0).abs()
    pivot["top10_share_gap_pp"] = pivot["female_top10_share"] - pivot["male_top10_share"]
    pivot["top5_share_gap_pp"] = pivot["female_top5_share"] - pivot["male_top5_share"]

    rows = []
    for outcome, odf in pivot.groupby("outcome", observed=True):
        _append_leaderboard(
            rows=rows,
            frame=odf,
            outcome=str(outcome),
            leaderboard="female_more_variable_raw",
            sort_col="raw_variance_ratio",
            ascending=False,
            top_k=top_k,
        )
        _append_leaderboard(
            rows=rows,
            frame=odf,
            outcome=str(outcome),
            leaderboard="male_more_variable_raw",
            sort_col="raw_variance_ratio",
            ascending=True,
            top_k=top_k,
        )
        _append_leaderboard(
            rows=rows,
            frame=odf,
            outcome=str(outcome),
            leaderboard="largest_residual_variance_gap",
            sort_col="residual_variance_gap_from_parity",
            ascending=False,
            top_k=top_k,
        )
        odf = odf.assign(__top10_gap_abs=odf["top10_share_gap_pp"].abs())
        _append_leaderboard(
            rows=rows,
            frame=odf,
            outcome=str(outcome),
            leaderboard="largest_top10_share_gap",
            sort_col="__top10_gap_abs",
            ascending=False,
            top_k=top_k,
        )

    if not rows:
        return pd.DataFrame(columns=OCCUPATION_LEADERS_COLUMNS)
    leaders = pd.DataFrame(rows).loc[:, OCCUPATION_LEADERS_COLUMNS]
    return leaders.sort_values(["outcome", "leaderboard", "rank"], ignore_index=True)


def _append_leaderboard(
    rows: list[dict],
    frame: pd.DataFrame,
    outcome: str,
    leaderboard: str,
    sort_col: str,
    ascending: bool,
    top_k: int,
) -> None:
    ranked = frame.dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending).head(top_k)
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        rows.append(
            {
                "leaderboard": leaderboard,
                "rank": rank,
                "outcome": outcome,
                "occupation_harmonized_code": str(getattr(row, "occupation_harmonized_code")),
                "occupation_harmonized_title": str(getattr(row, "occupation_harmonized_title")),
                "occupation_harmonization_type": str(
                    getattr(row, "occupation_harmonization_type")
                ),
                "occupation_title_vintage": str(getattr(row, "occupation_title_vintage")),
                "soc_major_group": str(getattr(row, "soc_major_group")),
                "soc_major_label": str(getattr(row, "soc_major_label")),
                "n_obs": int(getattr(row, "n_obs")),
                "raw_variance_ratio": float(getattr(row, "raw_variance_ratio")),
                "residual_variance_ratio": float(getattr(row, "residual_variance_ratio")),
                "female_p90_p10": float(getattr(row, "female_p90_p10")),
                "male_p90_p10": float(getattr(row, "male_p90_p10")),
                "female_top10_share": float(getattr(row, "female_top10_share")),
                "male_top10_share": float(getattr(row, "male_top10_share")),
                "female_top5_share": float(getattr(row, "female_top5_share")),
                "male_top5_share": float(getattr(row, "male_top5_share")),
                "raw_variance_gap_from_parity": float(
                    getattr(row, "raw_variance_gap_from_parity")
                ),
                "residual_variance_gap_from_parity": float(
                    getattr(row, "residual_variance_gap_from_parity")
                ),
                "top10_share_gap_pp": float(getattr(row, "top10_share_gap_pp")),
                "top5_share_gap_pp": float(getattr(row, "top5_share_gap_pp")),
            }
        )


def _summarize_harmonized_occupation_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "occupation_harmonized_code" not in frame.columns:
        return pd.DataFrame(columns=OCCUPATION_DISPERSION_COLUMNS[7:])

    grouped = (
        frame.loc[frame["occupation_harmonized_code"].notna()].groupby(
            "occupation_harmonized_code", observed=True, dropna=False
        )
    )
    summary = (
        grouped.apply(
            lambda df: pd.Series(
                {
                    "occupation_harmonized_title": _collapse_metadata_value(
                        df["occupation_harmonized_title"]
                    ),
                    "occupation_harmonization_type": _collapse_metadata_value(
                        df["occupation_harmonization_type"],
                        mixed_label="mixed_vintage_mapping",
                    ),
                    "occupation_title_vintage": _collapse_metadata_value(
                        df["occupation_title_vintage"], mixed_label="mixed"
                    ),
                    "soc_major_group": _collapse_metadata_value(
                        df["soc_major_group"], mixed_label="mixed"
                    ),
                    "soc_major_label": _collapse_metadata_value(
                        df["soc_major_label"], mixed_label="Mixed"
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"occupation_harmonized_code": "occupation_harmonized_code"})
    )
    for column in OCCUPATION_DISPERSION_COLUMNS[7:]:
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary.loc[:, OCCUPATION_DISPERSION_COLUMNS[7:]]


def _harmonize_occupation_surface(frame: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=OCCUPATION_DISPERSION_COLUMNS)
    out = frame.copy()
    required = [
        "occupation_harmonized_code",
        "occupation_harmonized_title",
        "occupation_harmonization_type",
    ]
    missing_required = [column for column in required if column not in out.columns]
    if strict and missing_required:
        raise ValueError(
            "Year-aware occupation harmonization metadata is required but missing: "
            + ", ".join(missing_required)
        )
    if "occupation_harmonized_code" not in out.columns:
        out["occupation_harmonized_code"] = out["stratum"].astype("string")
    if "occupation_harmonized_title" not in out.columns:
        out["occupation_harmonized_title"] = out.get(
            "occupation_title",
            pd.Series(pd.NA, index=out.index),
        )
    if "occupation_harmonization_type" not in out.columns:
        out["occupation_harmonization_type"] = pd.Series(
            "legacy_unscoped",
            index=out.index,
            dtype="string",
        )
    for col in [
        "occupation_title_vintage",
        "soc_major_group",
        "soc_major_label",
        "occupation_code_raw",
        "occupation_title_raw",
        "occupation_mapping_regime",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
    metadata = _summarize_harmonized_occupation_metadata(out)
    value_frame = (
        out.groupby(
            ["suite", "outcome", "metric", "occupation_harmonized_code"],
            observed=True,
            dropna=False,
        )
        .apply(
            lambda df: pd.Series(
                {
                    "stratifier": "occupation_harmonized_code",
                    "stratum": _collapse_metadata_value(df["occupation_harmonized_code"]),
                    "n_obs": int(pd.to_numeric(df["n_obs"], errors="coerce").sum()),
                    "value": _weighted_mean_by_n(df["value"], df["n_obs"]),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    value_frame = value_frame.merge(metadata, on="occupation_harmonized_code", how="left")
    for column in OCCUPATION_DISPERSION_COLUMNS:
        if column not in value_frame.columns:
            value_frame[column] = pd.NA
    return value_frame.loc[:, OCCUPATION_DISPERSION_COLUMNS]


def _attach_occupation_metadata_year_aware(
    frame: pd.DataFrame,
    code_col: str,
    survey_year_col: str,
    require_year_context: bool,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    frame = frame.copy()
    has_year_context = (
        survey_year_col in frame.columns
        and pd.to_numeric(frame[survey_year_col], errors="coerce").notna().any()
    )
    if require_year_context and not has_year_context:
        raise ValueError(
            "Year-aware occupation harmonization requires a populated survey year column."
        )
    if has_year_context:
        return attach_occupation_metadata(
            frame,
            code_col=code_col,
            survey_year_col=survey_year_col,
        )
    return attach_occupation_metadata(frame, code_col=code_col)


def _build_occupation_harmonization_map(occupation_df: pd.DataFrame) -> pd.DataFrame:
    if occupation_df.empty:
        return pd.DataFrame(columns=_occupation_harmonization_columns())
    columns = _occupation_harmonization_columns()
    if (
        "occupation_mapping_regime" not in occupation_df.columns
        and "survey_year_regime" in occupation_df.columns
    ):
        occupation_df["occupation_mapping_regime"] = occupation_df["survey_year_regime"]
    if "occupation_mapping_regime" not in occupation_df.columns:
        vintage = occupation_df.get(
            "occupation_title_vintage", pd.Series(pd.NA, index=occupation_df.index)
        )
        vintage_text = vintage.astype("string").str.lower()
        occupation_df["occupation_mapping_regime"] = np.select(
            [
                vintage_text.str.contains("2010", na=False),
                vintage_text.str.contains("2018", na=False),
            ],
            ["pre_2018", "post_2018"],
            default="unknown",
        )
    for col in columns:
        if col not in occupation_df.columns:
            occupation_df[col] = pd.NA
    mapped = (
        occupation_df.loc[:, columns]
        .drop_duplicates()
        .sort_values(
            ["occupation_harmonized_code", "occupation_mapping_regime", "occupation_code_raw"],
            ignore_index=True,
        )
    )
    return mapped


def _collapse_metadata_value(series: pd.Series, mixed_label: str = "mixed"):
    values = series.astype("string")
    values = values.loc[values.notna() & values.ne("")].drop_duplicates()
    if values.empty:
        return pd.NA
    if len(values) == 1:
        return values.iloc[0]
    return mixed_label


def _occupation_harmonization_columns() -> list[str]:
    return [
        "occupation_code_raw",
        "occupation_title_raw",
        "occupation_title_vintage",
        "occupation_mapping_regime",
        "occupation_harmonized_code",
        "occupation_harmonized_title",
        "occupation_harmonization_type",
        "soc_major_group",
        "soc_major_label",
    ]


def _weighted_mean_by_n(values: pd.Series, weights: pd.Series) -> float:
    val = pd.to_numeric(values, errors="coerce")
    wgt = pd.to_numeric(weights, errors="coerce")
    mask = val.notna() & wgt.notna() & wgt.gt(0)
    if not mask.any():
        return float("nan")
    return float(np.average(val.loc[mask], weights=wgt.loc[mask]))


def _build_tail_contrast_summary(results_dir: Path) -> pd.DataFrame:
    output_columns = [
        "suite",
        "outcome",
        "stratifier",
        "stratum",
        "n_obs",
        "raw_variance_ratio",
        "residual_variance_ratio",
        "female_top10_share",
        "male_top10_share",
        "female_top5_share",
        "male_top5_share",
        "top10_share_gap_pp",
        "top5_share_gap_pp",
        "female_to_male_top10_ratio",
        "female_to_male_top5_ratio",
    ]
    frames = []
    for file_name in [
        "acs_variance_suite.csv",
        "acs_reproductive_dispersion.csv",
        "acs_onet_dispersion.csv",
    ]:
        path = results_dir / file_name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=output_columns)
    base = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["suite", "outcome", "stratifier", "stratum", "metric"], keep="last"
    )
    pivot = base.pivot_table(
        index=["suite", "outcome", "stratifier", "stratum", "n_obs"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    for col in [
        "raw_variance_ratio",
        "residual_variance_ratio",
        "female_top10_share",
        "male_top10_share",
        "female_top5_share",
        "male_top5_share",
    ]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["female_to_male_top10_ratio"] = pivot["female_top10_share"] / pivot["male_top10_share"]
    pivot["female_to_male_top5_ratio"] = pivot["female_top5_share"] / pivot["male_top5_share"]
    pivot["top10_share_gap_pp"] = pivot["female_top10_share"] - pivot["male_top10_share"]
    pivot["top5_share_gap_pp"] = pivot["female_top5_share"] - pivot["male_top5_share"]
    for column in output_columns:
        if column not in pivot.columns:
            pivot[column] = np.nan
    return pivot.loc[:, output_columns]


def _build_year_regime_variance_summary(results_dir: Path) -> pd.DataFrame:
    output_columns = ["suite", "outcome", "metric", "regime", "weighted_mean_value", "n_obs_total"]
    frames = []
    for file_name in ["acs_variance_suite.csv", "acs_selection_corrected_variance.csv"]:
        path = results_dir / file_name
        if path.exists():
            frame = pd.read_csv(path)
            if "suite" not in frame.columns:
                frame["suite"] = "variance_addon"
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=output_columns)
    df = pd.concat(frames, ignore_index=True)
    year = pd.to_numeric(df["stratum"], errors="coerce")
    df = df.loc[df["stratifier"] == "survey_year"].copy()
    df["survey_year"] = year.loc[df.index]
    regime_years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
    df = df.loc[df["survey_year"].isin(regime_years)].copy()
    df["regime"] = np.where(df["survey_year"] <= 2019, "pre_2020", "post_2020")
    grouped = (
        df.groupby(["suite", "outcome", "metric", "regime"], observed=True)
        .apply(
            lambda frame: pd.Series(
                {
                    "weighted_mean_value": _weighted_mean_by_n(frame["value"], frame["n_obs"]),
                    "n_obs_total": int(pd.to_numeric(frame["n_obs"], errors="coerce").sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    for column in output_columns:
        if column not in grouped.columns:
            grouped[column] = np.nan
    return grouped.loc[:, output_columns]


def _build_soc_group_leaderboard_counts(results_dir: Path) -> pd.DataFrame:
    leaders_path = results_dir / "acs_occupation_variability_leaders.csv"
    if not leaders_path.exists():
        return pd.DataFrame(
            columns=[
                "outcome",
                "leaderboard",
                "soc_major_group",
                "soc_major_label",
                "top10_count",
            ]
        )
    leaders = pd.read_csv(leaders_path)
    if leaders.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "leaderboard",
                "soc_major_group",
                "soc_major_label",
                "top10_count",
            ]
        )
    top10 = leaders.loc[pd.to_numeric(leaders["rank"], errors="coerce") <= 10].copy()
    counts = (
        top10.groupby(
            ["outcome", "leaderboard", "soc_major_group", "soc_major_label"], observed=True
        )
        .size()
        .rename("top10_count")
        .reset_index()
        .sort_values(["outcome", "leaderboard", "top10_count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return counts


def _build_fertility_risk_variance_bridge(results_dir: Path) -> pd.DataFrame:
    quartile_path = results_dir / "acs_fertility_risk_by_quartile.csv"
    if not quartile_path.exists():
        quartile_path = PROJECT_ROOT / "results" / "repro" / "acs_fertility_risk_by_quartile.csv"
    penalty_path = results_dir / "acs_fertility_risk_penalty.csv"
    if not penalty_path.exists():
        penalty_path = PROJECT_ROOT / "results" / "repro" / "acs_fertility_risk_penalty.csv"
    repro_dispersion_path = results_dir / "acs_reproductive_dispersion.csv"
    if not repro_dispersion_path.exists():
        repro_dispersion_path = (
            PROJECT_ROOT / "results" / "variance" / "acs_reproductive_dispersion.csv"
        )

    if (
        not quartile_path.exists()
        or not penalty_path.exists()
        or not repro_dispersion_path.exists()
    ):
        return pd.DataFrame(
            columns=[
                "normalized_outcome",
                "risk_quartile",
                "mean_outcome",
                "n_obs",
                "weighted_n",
                "raw_variance_ratio",
                "residual_variance_ratio",
                "female_top10_share",
                "male_top10_share",
                "female_to_male_top10_ratio",
                "female_penalty_coef",
                "female_penalty_n_obs",
            ]
        )

    quartile = pd.read_csv(quartile_path)
    penalty = pd.read_csv(penalty_path)
    dispersion = pd.read_csv(repro_dispersion_path)

    quartile["normalized_outcome"] = (
        quartile["outcome"].astype("string").map(_normalize_outcome_name)
    )
    quartile["risk_quartile"] = quartile["risk_quartile"].astype("string")
    quartile = quartile.loc[
        :, ["normalized_outcome", "risk_quartile", "mean_outcome", "n_obs", "weighted_n"]
    ]

    penalty = penalty.loc[penalty["term"] == "female"].copy()
    penalty["normalized_outcome"] = (
        penalty["outcome"].astype("string").map(_normalize_outcome_name)
    )
    penalty_summary = (
        penalty.groupby(["normalized_outcome"], observed=True)
        .apply(
            lambda d: pd.Series(
                {
                    "female_penalty_coef": _weighted_mean_by_n(d["coef"], d["n_obs"]),
                    "female_penalty_n_obs": int(pd.to_numeric(d["n_obs"], errors="coerce").sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    dispersion = dispersion.loc[dispersion["stratifier"] == "fertility_risk_quartile"].copy()
    dispersion["normalized_outcome"] = (
        dispersion["outcome"].astype("string").map(_normalize_outcome_name)
    )
    dispersion_pivot = (
        dispersion.pivot_table(
            index=["normalized_outcome", "stratum", "n_obs"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"stratum": "risk_quartile", "n_obs": "dispersion_n_obs"})
    )
    for col in [
        "raw_variance_ratio",
        "residual_variance_ratio",
        "female_top10_share",
        "male_top10_share",
    ]:
        if col not in dispersion_pivot.columns:
            dispersion_pivot[col] = np.nan
    dispersion_pivot["female_to_male_top10_ratio"] = (
        dispersion_pivot["female_top10_share"] / dispersion_pivot["male_top10_share"]
    )

    bridge = quartile.merge(
        dispersion_pivot[
            [
                "normalized_outcome",
                "risk_quartile",
                "raw_variance_ratio",
                "residual_variance_ratio",
                "female_top10_share",
                "male_top10_share",
                "female_to_male_top10_ratio",
            ]
        ],
        on=["normalized_outcome", "risk_quartile"],
        how="left",
    ).merge(
        penalty_summary,
        on="normalized_outcome",
        how="left",
    )
    return bridge


def _normalize_outcome_name(outcome: str) -> str:
    text = str(outcome).lower()
    if "hourly" in text:
        return "hourly"
    if "annual" in text:
        return "annual"
    return text
