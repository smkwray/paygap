"""Top-level reproductive-burden extension runner."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from gender_gap.features.household import enrich_household_features
from gender_gap.features.occupation_context import build_onet_indices, merge_onet_context
from gender_gap.features.reproductive import (
    add_fertility_risk_features,
    add_repro_interactions,
    add_reproductive_features,
)
from gender_gap.features.sample_filters import filter_all_employed, filter_prime_age_wage_salary
from gender_gap.models.atus_mechanisms import build_atus_mechanism_table
from gender_gap.models.descriptive import (
    build_lesbian_married_adjusted_table,
    build_lesbian_married_summary,
)
from gender_gap.models.fertility_risk import build_same_sex_placebos, run_fertility_risk_penalty
from gender_gap.models.ols import (
    BLOCK_DEFINITIONS,
    coefficient_table,
    required_columns_for_model,
    results_to_dataframe,
    run_sequential_ols,
)
from gender_gap.models.sipp_robustness import build_sipp_robustness_table
from gender_gap.models.variance_suite import run_variance_suite
from gender_gap.reporting.repro import (
    build_local_inventory_report,
    build_optional_validation_status,
    build_repro_inventory_usage,
    build_repro_release_manifest,
    validate_repro_output_schemas,
    write_atus_mechanisms_report,
    write_local_inventory_report,
    write_nlsy_validation_output,
    write_repro_inventory_report,
    write_repro_release_manifest,
    write_repro_schema_check,
    write_repro_summary,
)
from gender_gap.settings import PROJECT_ROOT, load_repro_config, shared_source_path
from gender_gap.standardize.acs_standardize import standardize_acs
from gender_gap.utils.io import read_parquet, write_parquet

logger = logging.getLogger(__name__)

PANEL_BASE_COLUMNS = {
    "survey_year",
    "female",
    "age",
    "age_sq",
    "race_ethnicity",
    "education_level",
    "state_fips",
    "occupation_code",
    "industry_code",
    "class_of_worker",
    "usual_hours_week",
    "work_from_home",
    "commute_minutes_one_way",
    "marital_status",
    "number_children",
    "children_under_5",
    "hourly_wage_real",
    "annual_earnings_real",
    "log_hourly_wage_real",
    "log_annual_earnings_real",
    "person_weight",
    "recent_birth",
    "recent_marriage",
    "has_own_child",
    "own_child_under6",
    "own_child_6_17_only",
    "same_sex_couple_household",
    "couple_type",
    "reproductive_stage",
    "employment_indicator",
    "ftfy_indicator",
}

PANEL_FINAL_EXTRA_COLUMNS = {
    "autonomy",
    "schedule_unpredictability",
    "time_pressure",
    "coordination_responsibility",
    "physical_proximity",
    "job_rigidity",
    "job_rigidity_quartile",
    "fertility_risk_score",
    "fertility_risk_quartile",
    "female_x_recent_birth",
    "female_x_own_child_under6",
    "female_x_recent_marriage",
    "female_x_same_sex_couple_household",
    "female_x_autonomy",
    "female_x_job_rigidity",
    "female_x_own_child_under6_x_job_rigidity",
}

PANEL_HOUSEHOLD_COLUMNS = {
    "partner_wage_real",
    "partner_employed",
    "multigenerational",
    "other_adults_present",
}

CATEGORICAL_PANEL_COLUMNS = [
    "race_ethnicity",
    "education_level",
    "marital_status",
    "couple_type",
    "reproductive_stage",
    "fertility_risk_quartile",
    "job_rigidity_quartile",
]

INTEGER_PANEL_COLUMNS = [
    "survey_year",
    "female",
    "age",
    "state_fips",
    "occupation_code",
    "industry_code",
    "class_of_worker",
    "number_children",
    "children_under_5",
    "recent_birth",
    "recent_marriage",
    "has_own_child",
    "own_child_under6",
    "own_child_6_17_only",
    "same_sex_couple_household",
    "work_from_home",
    "employment_indicator",
    "ftfy_indicator",
    "partner_employed",
    "multigenerational",
    "other_adults_present",
]


def run_repro_extension() -> dict[str, Path]:
    """Run the public-core reproductive-burden extension using available inputs."""
    cfg = load_repro_config()
    acs_years = cfg.get("datasets", {}).get("acs", {}).get("years", [])
    required_onet = cfg.get("datasets", {}).get("onet", {}).get("required_files", [])
    paths = cfg.get("paths", {})
    analysis_sample = cfg.get("analysis", {}).get("sample", "prime_age_wage_salary")

    results_dir = PROJECT_ROOT / paths.get("results_dir", "results/repro")
    diagnostics_dir = PROJECT_ROOT / paths.get("diagnostics_dir", "results/diagnostics")
    report_path = PROJECT_ROOT / paths.get("report_path", "reports/repro_extension_summary.md")
    panel_output = PROJECT_ROOT / paths.get("acs_panel_output", "data/processed/acs_repro_panel.parquet")
    onet_output = PROJECT_ROOT / paths.get("onet_indices_output", "data/processed/onet_indices.parquet")
    onet_coverage_output = PROJECT_ROOT / paths.get("onet_merge_coverage", "results/diagnostics/onet_merge_coverage.csv")
    inventory_output = PROJECT_ROOT / paths.get("inventory_usage_output", "results/diagnostics/repro_inventory_usage.csv")
    inventory_report_output = PROJECT_ROOT / paths.get("inventory_usage_report", "reports/repro_inventory_usage.md")
    optional_validation_output = PROJECT_ROOT / paths.get(
        "optional_validation_output",
        "results/diagnostics/repro_optional_validation_status.csv",
    )
    inventory_config = PROJECT_ROOT / paths.get("inventory_config", "inventory/inventory_paths.yaml")
    local_inventory_output = PROJECT_ROOT / paths.get(
        "local_inventory_output",
        "diagnostics/local_inventory_report.json",
    )
    atus_report_output = PROJECT_ROOT / paths.get(
        "atus_report_output",
        "reports/atus_repro_mechanisms.md",
    )
    nlsy_validation_output = PROJECT_ROOT / paths.get(
        "nlsy_validation_output",
        "results/repro/nlsy_validation.csv",
    )
    release_manifest_output = PROJECT_ROOT / paths.get(
        "release_manifest_output",
        "diagnostics/repro_release_manifest.json",
    )
    schema_snapshot_path = PROJECT_ROOT / paths.get(
        "schema_snapshot_path",
        "configs/repro_output_schemas.json",
    )
    schema_check_output = PROJECT_ROOT / paths.get(
        "schema_check_output",
        "diagnostics/repro_schema_check.json",
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    inventory_usage = build_repro_inventory_usage(acs_years, required_onet)
    inventory_output.parent.mkdir(parents=True, exist_ok=True)
    inventory_usage.to_csv(inventory_output, index=False)
    write_repro_inventory_report(inventory_usage, inventory_report_output)
    optional_validation = build_optional_validation_status()
    optional_validation.to_csv(optional_validation_output, index=False)
    outputs: dict[str, Path] = {
        "inventory_usage": inventory_output,
        "inventory_report": inventory_report_output,
        "optional_validation_status": optional_validation_output,
    }

    missing_inputs = inventory_usage.loc[inventory_usage["status"] == "missing", "asset_name"].tolist()
    pooled_frames = []
    available_years: list[int] = []
    for year in acs_years:
        frame = _load_acs_year(year)
        if frame is None or frame.empty:
            continue
        frame = _apply_analysis_sample(frame, analysis_sample)
        if frame.empty:
            continue
        available_years.append(year)
        pooled_frames.append(_compress_panel(frame, PANEL_BASE_COLUMNS | PANEL_HOUSEHOLD_COLUMNS))

    if not pooled_frames:
        raise FileNotFoundError("No ACS annual files available for the reproductive extension")

    panel = pd.concat(pooled_frames, ignore_index=True)
    panel = add_reproductive_features(panel)

    onet_coverage = pd.DataFrame()
    onet_dir = _resolve_onet_dir(required_onet)
    try:
        onet_indices = build_onet_indices(
            onet_dir=onet_dir,
            recipe_path=PROJECT_ROOT / "configs" / "onet_index_recipe.yaml",
        )
        write_parquet(onet_indices, onet_output)
        panel, onet_coverage = merge_onet_context(panel, onet_indices)
        onet_coverage.to_csv(onet_coverage_output, index=False)
    except FileNotFoundError as exc:
        missing_inputs.append(str(exc))
        logger.warning("O*NET merge skipped: %s", exc)

    panel = add_fertility_risk_features(panel)
    panel = add_repro_interactions(panel)
    panel = _compress_panel(panel, _final_panel_columns())
    write_parquet(panel, panel_output)

    ols = run_sequential_ols(
        panel,
        weight_col="person_weight",
        blocks={k: v for k, v in BLOCK_DEFINITIONS.items() if k.startswith("M")},
    )
    ols_path = results_dir / "acs_gap_ladder_extended.csv"
    results_to_dataframe(ols).to_csv(ols_path, index=False)
    outputs["gap_ladder"] = ols_path

    yearly_rows = []
    for year, ydf in panel.groupby("survey_year", observed=True):
        y_ols = run_sequential_ols(
            ydf,
            weight_col="person_weight",
            blocks={k: v for k, v in BLOCK_DEFINITIONS.items() if k.startswith("M")},
        )
        year_df = results_to_dataframe(y_ols)
        year_df.insert(0, "year", year)
        yearly_rows.append(year_df)
    by_year_path = results_dir / "acs_gap_ladder_by_year.csv"
    pd.concat(yearly_rows, ignore_index=True).to_csv(by_year_path, index=False)
    outputs["gap_ladder_by_year"] = by_year_path

    interaction_path = results_dir / "acs_onet_interactions.csv"
    coefficient_table(
        panel,
        model_name="M8_reproductive_x_job_context",
        weight_col="person_weight",
    ).loc[lambda d: d["term"].str.contains("female_x_")].to_csv(interaction_path, index=False)
    outputs["onet_interactions"] = interaction_path

    penalty_df, quartile_df = run_fertility_risk_penalty(panel)
    penalty_path = results_dir / "acs_fertility_risk_penalty.csv"
    penalty_df.to_csv(penalty_path, index=False)
    outputs["fertility_penalty"] = penalty_path

    quartile_path = results_dir / "acs_fertility_risk_by_quartile.csv"
    quartile_df.to_csv(quartile_path, index=False)
    outputs["fertility_quartiles"] = quartile_path

    placebo_path = results_dir / "acs_same_sex_placebos.csv"
    build_same_sex_placebos(panel).to_csv(placebo_path, index=False)
    outputs["same_sex_placebos"] = placebo_path

    lesbian_married_path = results_dir / "acs_lesbian_married_subgroup.csv"
    build_lesbian_married_summary(panel).to_csv(lesbian_married_path, index=False)
    outputs["lesbian_married_subgroup"] = lesbian_married_path

    lesbian_adjusted_path = results_dir / "acs_lesbian_married_adjusted.csv"
    build_lesbian_married_adjusted_table(panel).to_csv(lesbian_adjusted_path, index=False)
    outputs["lesbian_married_adjusted"] = lesbian_adjusted_path

    variance_df = run_variance_suite(panel)
    variance_path = results_dir / "acs_variance_suite.csv"
    variance_df.to_csv(variance_path, index=False)
    outputs["variance_suite"] = variance_path

    household_sensitivity = _build_household_sensitivity(panel)
    household_sensitivity_path = results_dir / "acs_household_sensitivity.csv"
    household_sensitivity.to_csv(household_sensitivity_path, index=False)
    outputs["household_sensitivity"] = household_sensitivity_path

    tail_path = results_dir / "acs_tail_metrics.csv"
    if "metric" in variance_df.columns:
        variance_df.loc[
            variance_df["metric"].str.contains("top|p90|p95", case=False, regex=True, na=False)
        ].to_csv(tail_path, index=False)
    else:
        pd.DataFrame(columns=["stratifier", "stratum", "metric", "value", "n_obs"]).to_csv(
            tail_path,
            index=False,
        )
    outputs["tail_metrics"] = tail_path

    atus_status, atus_path = _run_atus_mechanisms(results_dir)
    outputs["atus_mechanisms"] = atus_path
    sipp_status, sipp_path = _run_sipp_robustness(results_dir)
    outputs["sipp_robustness"] = sipp_path
    atus_report_path = write_atus_mechanisms_report(atus_path, atus_report_output)
    outputs["atus_report"] = atus_report_path

    nlsy_validation_path = write_nlsy_validation_output(nlsy_validation_output)
    if nlsy_validation_path is not None:
        outputs["nlsy_validation"] = nlsy_validation_path

    if inventory_config.exists():
        local_inventory_report = build_local_inventory_report(inventory_config)
        write_local_inventory_report(local_inventory_report, local_inventory_output)
        outputs["local_inventory_report_json"] = local_inventory_output
        outputs["local_inventory_report_md"] = local_inventory_output.with_suffix(".md")

    write_repro_summary(
        output_path=report_path,
        available_years=available_years,
        inventory_usage=inventory_usage,
        missing_inputs=missing_inputs,
        generated_files=list(outputs.values()),
        onet_coverage=onet_coverage,
        atus_status=atus_status,
        sipp_status=sipp_status,
        optional_validation=optional_validation,
    )
    outputs["summary_report"] = report_path

    release_manifest = build_repro_release_manifest(
        output_paths=list(outputs.values()),
        inventory_usage=inventory_usage,
        optional_validation=optional_validation,
    )
    write_repro_release_manifest(release_manifest, release_manifest_output)
    outputs["release_manifest_json"] = release_manifest_output
    outputs["release_manifest_md"] = release_manifest_output.with_suffix(".md")

    if schema_snapshot_path.exists():
        schema_report = validate_repro_output_schemas(schema_snapshot_path)
        write_repro_schema_check(schema_report, schema_check_output)
        outputs["schema_check_json"] = schema_check_output
        outputs["schema_check_md"] = schema_check_output.with_suffix(".md")

    return outputs


def _load_acs_year(year: int) -> pd.DataFrame | None:
    raw = _resolve_acs_raw_path(year)
    if raw is None:
        processed = PROJECT_ROOT / "data" / "processed" / f"acs_{year}_analysis_ready_repweights.parquet"
        if processed.exists():
            df = read_parquet(processed)
            df = _drop_replicate_weights(df)
            return add_reproductive_features(enrich_household_features(df))
        return None
    raw_df = read_parquet(raw, columns=_available_columns(raw, _acs_raw_columns(year)))
    standardized = standardize_acs(raw_df, survey_year=year, keep_replicate_weights=False)
    standardized = enrich_household_features(standardized)
    return add_reproductive_features(standardized)


def _resolve_acs_raw_path(year: int) -> Path | None:
    filename = f"acs_pums_{year}_api_repweights.parquet"
    shared = shared_source_path("census", "acs", "wave2", "paygap", "raw", "acs", filename)
    if shared.exists():
        return shared
    local = PROJECT_ROOT / "data" / "raw" / "acs" / filename
    if local.exists():
        return local
    return None


def _resolve_onet_dir(required_files: list[str]) -> Path:
    shared = shared_source_path("onet", "db_30_2_text")
    if _dir_has_files(shared, required_files):
        return shared
    local = PROJECT_ROOT / "data" / "raw" / "context" / "onet" / "db_30_2_text"
    if _dir_has_files(local, required_files):
        return local
    return local


def _dir_has_files(path: Path, required_files: list[str]) -> bool:
    return path.exists() and all((path / filename).exists() for filename in required_files)


def _acs_raw_columns(year: int) -> list[str]:
    columns = [
        "SERIALNO",
        "SPORDER",
        "SEX",
        "AGEP",
        "SCHL",
        "HISP",
        "RAC1P",
        "WAGP",
        "PERNP",
        "WKHP",
        "WKW",
        "WKWN",
        "COW",
        "OCCP",
        "INDP",
        "JWTR",
        "JWTRNS",
        "JWMNP",
        "ST",
        "STATE",
        "PUMA",
        "PWGTP",
        "MAR",
        "ADJINC",
        "NOC",
        "PAOC",
        "FER",
        "MARHM",
        "PARTNER",
        "POWSP",
        "POWPUMA",
        "MULTG",
    ]
    if year >= 2019:
        columns.extend(["CPLT", "RELSHIPP"])
    else:
        columns.append("RELP")
    return columns


def _drop_replicate_weights(df: pd.DataFrame) -> pd.DataFrame:
    keep = [col for col in df.columns if not col.startswith("rep_weight_")]
    return df.loc[:, keep].copy()


def _available_columns(path: Path, requested: list[str]) -> list[str]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    return [column for column in requested if column in available]


def _final_panel_columns() -> set[str]:
    required = set(required_columns_for_model("M8_reproductive_x_job_context"))
    required.update(PANEL_BASE_COLUMNS)
    required.update(PANEL_FINAL_EXTRA_COLUMNS)
    required.update(PANEL_HOUSEHOLD_COLUMNS)
    return required


def _build_household_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    base_controls = list(BLOCK_DEFINITIONS["M7_onet_context"])
    composition_terms = _available_household_terms(
        panel,
        ["multigenerational", "other_adults_present"],
    )
    partner_resource_terms = _available_household_terms(
        panel,
        ["partner_employed", "partner_wage_real"],
    )
    specifications = [
        (
            "household_composition",
            "full_sample",
            base_controls,
            base_controls + composition_terms,
            "M7_onet_context",
            "M7_onet_context_plus_household_composition",
        ),
        (
            "partner_resources",
            "partnered_households",
            base_controls,
            base_controls + partner_resource_terms,
            "M7_onet_context_partnered_baseline",
            "M7_onet_context_plus_partner_resources",
        ),
    ]

    frames: list[pd.DataFrame] = []
    for panel_name, sample_name, base_terms, augmented_terms, baseline_name, augmented_name in specifications:
        blocks = {
            baseline_name: base_terms,
            augmented_name: augmented_terms,
        }
        subset = _matched_sample_for_blocks(
            panel,
            blocks=blocks,
            outcome="log_hourly_wage_real",
            weight_col="person_weight",
        )
        results = results_to_dataframe(
            run_sequential_ols(
                subset,
                weight_col="person_weight",
                blocks=blocks,
            )
        )
        results.insert(0, "sample", sample_name)
        results.insert(0, "panel", panel_name)
        frames.append(results)

    return pd.concat(frames, ignore_index=True)


def _matched_sample_for_blocks(
    df: pd.DataFrame,
    blocks: dict[str, list[str]],
    outcome: str,
    weight_col: str,
) -> pd.DataFrame:
    required_columns: list[str] = []
    substitute_hourly_outcome = outcome == "log_hourly_wage_real" and outcome not in df.columns
    for model_name in blocks:
        for column in required_columns_for_model(
            model_name,
            outcome=outcome,
            weight_col=weight_col,
            blocks=blocks,
        ):
            if substitute_hourly_outcome and column == outcome:
                column = "hourly_wage_real"
            if column not in required_columns:
                required_columns.append(column)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns for household sensitivity: "
            + ", ".join(sorted(missing_columns))
        )

    subset = df.loc[:, required_columns].copy()
    return subset.dropna(subset=required_columns)


def _available_household_terms(df: pd.DataFrame, terms: list[str]) -> list[str]:
    return [
        term
        for term in terms
        if term in df.columns and pd.to_numeric(df[term], errors="coerce").notna().any()
    ]


def _compress_panel(df: pd.DataFrame, keep_columns: set[str]) -> pd.DataFrame:
    keep = [column for column in df.columns if column in keep_columns]
    compressed = df.loc[:, keep].copy()

    for column in CATEGORICAL_PANEL_COLUMNS:
        if column in compressed.columns:
            compressed[column] = compressed[column].astype("category")

    for column in INTEGER_PANEL_COLUMNS:
        if column in compressed.columns:
            numeric = pd.to_numeric(compressed[column], errors="coerce")
            if numeric.notna().any():
                compressed[column] = pd.to_numeric(numeric, downcast="integer")

    float_columns = [
        column
        for column in compressed.columns
        if column not in CATEGORICAL_PANEL_COLUMNS and column not in INTEGER_PANEL_COLUMNS
    ]
    for column in float_columns:
        numeric = pd.to_numeric(compressed[column], errors="coerce")
        if numeric.notna().any():
            compressed[column] = pd.to_numeric(numeric, downcast="float")

    return compressed


def _apply_analysis_sample(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    if sample_name in {"prime_age", "prime_age_wage_salary"}:
        return filter_prime_age_wage_salary(df)
    if sample_name == "all_employed":
        return filter_all_employed(df)
    logger.warning("Unknown analysis sample '%s'; leaving frame unfiltered", sample_name)
    return df


def _run_atus_mechanisms(results_dir: Path) -> tuple[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "atus_mechanisms.csv"
    atus_processed = PROJECT_ROOT / "data" / "processed" / "atus_analysis_ready.parquet"
    atus_respondent = PROJECT_ROOT / "data" / "raw" / "atus" / "atus_respondent.parquet"
    atus_roster = PROJECT_ROOT / "data" / "raw" / "atus" / "atus_roster.parquet"
    missing = [p.name for p in [atus_processed, atus_respondent, atus_roster] if not p.exists()]
    if missing:
        pd.DataFrame([{"status": "skipped", "reason": f"ATUS inputs missing: {', '.join(missing)}"}]).to_csv(path, index=False)
        return (f"skipped: missing ATUS inputs ({', '.join(missing)})", path)

    mechanisms = build_atus_mechanism_table(
        processed_path=atus_processed,
        respondent_path=atus_respondent,
        roster_path=atus_roster,
    )
    mechanisms.to_csv(path, index=False)
    stage_count = int(mechanisms["reproductive_stage"].nunique(dropna=True))
    return (f"ok: {len(mechanisms)} ATUS mechanism rows across {stage_count} stages", path)


def _run_sipp_robustness(results_dir: Path) -> tuple[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "sipp_robustness.csv"
    sipp_processed = PROJECT_ROOT / "data" / "processed" / "sipp_standardized.parquet"
    sipp_raw = PROJECT_ROOT / "data" / "raw" / "sipp" / "pu2024.dta"
    missing = [p.name for p in [sipp_processed, sipp_raw] if not p.exists()]
    if missing:
        pd.DataFrame([{"status": "skipped", "reason": f"SIPP inputs missing: {', '.join(missing)}"}]).to_csv(path, index=False)
        return (f"skipped: missing SIPP inputs ({', '.join(missing)})", path)
    table = build_sipp_robustness_table(
        standardized_path=sipp_processed,
        raw_path=sipp_raw,
        survey_year=2023,
    )
    table.to_csv(path, index=False)
    if table.empty:
        return ("skipped: empty SIPP robustness output", path)
    status = str(table["status"].iloc[0]) if "status" in table.columns else "ok_partial"
    return (f"{status}: {len(table)} SIPP robustness rows", path)
