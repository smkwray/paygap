"""Tests for the dedicated variance addon backend surface."""

import json

import numpy as np
import pandas as pd

from gender_gap import variance
from gender_gap.reporting import variance as variance_reporting


def test_variance_schema_helpers_roundtrip(tmp_path, monkeypatch):
    project_root = tmp_path / "paygap"
    results_dir = project_root / "results" / "variance"
    results_dir.mkdir(parents=True)
    csv_path = results_dir / "acs_variance_suite.csv"
    occ_path = results_dir / "acs_occupation_dispersion.csv"
    leaders_path = results_dir / "acs_occupation_variability_leaders.csv"
    pd.DataFrame(
        [
            {
                "suite": "V1_raw_residual",
                "outcome": "log_hourly_wage_real",
                "stratifier": "overall",
                "stratum": "all",
                "metric": "raw_variance_ratio",
                "value": 1.1,
                "n_obs": 100,
            }
        ]
    ).to_csv(csv_path, index=False)
    pd.DataFrame(
        [
            {
                "suite": "V5_occupation_dispersion",
                "outcome": "log_hourly_wage_real",
                "stratifier": "occupation_harmonized_code",
                "stratum": "1010",
                "metric": "raw_variance_ratio",
                "value": 1.05,
                "n_obs": 80,
                "occupation_harmonized_code": "1010",
                "occupation_harmonized_title": "Software developers",
                "occupation_harmonization_type": "native_2018",
                "occupation_title_vintage": "2018",
                "soc_major_group": "15",
                "soc_major_label": "Computer and Mathematical",
            }
        ]
    ).to_csv(occ_path, index=False)
    pd.DataFrame(
        [
            {
                "leaderboard": "female_more_variable_raw",
                "rank": 1,
                "outcome": "log_hourly_wage_real",
                "occupation_harmonized_code": "1010",
                "occupation_harmonized_title": "Software developers",
                "occupation_harmonization_type": "native_2018",
                "occupation_title_vintage": "2018",
                "soc_major_group": "15",
                "soc_major_label": "Computer and Mathematical",
                "n_obs": 80,
                "raw_variance_ratio": 1.05,
                "residual_variance_ratio": 1.02,
                "female_p90_p10": 1.4,
                "male_p90_p10": 1.2,
                "female_top10_share": 0.09,
                "male_top10_share": 0.1,
                "female_top5_share": 0.04,
                "male_top5_share": 0.05,
                "raw_variance_gap_from_parity": 0.05,
                "residual_variance_gap_from_parity": 0.02,
                "top10_share_gap_pp": -0.01,
                "top5_share_gap_pp": -0.01,
            }
        ]
    ).to_csv(leaders_path, index=False)
    schema_path = project_root / "configs" / "variance_output_schemas.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(
        json.dumps(
            {
                "files": {
                    "results/variance/acs_variance_suite.csv": list(
                        pd.read_csv(csv_path, nrows=0).columns
                    ),
                    "results/variance/acs_occupation_dispersion.csv": list(
                        pd.read_csv(occ_path, nrows=0).columns
                    ),
                    "results/variance/acs_occupation_variability_leaders.csv": list(
                        pd.read_csv(leaders_path, nrows=0).columns
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(variance_reporting, "PROJECT_ROOT", project_root)

    report = variance_reporting.validate_variance_output_schemas(schema_path)
    output = project_root / "diagnostics" / "variance_schema_check.json"
    variance_reporting.write_variance_schema_check(report, output)

    assert report["passed"] is True
    assert json.loads(output.read_text())["matched_files"] == 3
    assert output.with_suffix(".md").exists()


def test_write_variance_summary_renders_occupation_leaders(tmp_path):
    output_path = tmp_path / "variance_addon_summary.md"
    inventory_usage = pd.DataFrame(
        [
            {
                "asset_group": "acs",
                "asset_name": "acs_pums_2023_api_repweights.parquet",
                "status": "present",
                "legacy_path": "data/raw/acs/acs_pums_2023_api_repweights.parquet",
                "canonical_path": (
                    "sources/census/acs/wave2/paygap/raw/acs/acs_pums_2023_api_repweights.parquet"
                ),
                "note": "variance_addon",
            }
        ]
    )
    leaders = pd.DataFrame(
        [
            {
                "leaderboard": "female_more_variable_raw",
                "rank": 1,
                "outcome": "log_hourly_wage_real",
                "occupation_harmonized_code": 1010,
                "occupation_harmonized_title": "Software developers",
                "occupation_harmonization_type": "native_2018",
                "occupation_title_vintage": "2018",
                "soc_major_group": "15",
                "soc_major_label": "Computer and Mathematical",
                "n_obs": 540,
                "raw_variance_ratio": 1.12,
                "residual_variance_ratio": 1.07,
                "female_p90_p10": 1.5,
                "male_p90_p10": 1.2,
                "female_top10_share": 0.11,
                "male_top10_share": 0.08,
                "female_top5_share": 0.05,
                "male_top5_share": 0.04,
                "raw_variance_gap_from_parity": 0.12,
                "residual_variance_gap_from_parity": 0.07,
                "top10_share_gap_pp": 0.03,
                "top5_share_gap_pp": 0.01,
            }
        ]
    )

    variance_reporting.write_variance_summary(
        output_path=output_path,
        available_years=[2023],
        inventory_usage=inventory_usage,
        reused_outputs=[],
        addon_outputs=[],
        missing_inputs=[],
        occupation_leaders=leaders,
    )
    text = output_path.read_text(encoding="utf-8")
    assert "## Occupation-level variability leaders" in text
    assert "log_hourly_wage_real / female_more_variable_raw #1" in text
    assert "Software developers" in text
    assert "raw_ratio=1.12" in text


def test_run_variance_addon_promotes_repro_outputs_without_cached_panel(tmp_path, monkeypatch):
    project_root = tmp_path / "paygap"
    (project_root / "results" / "repro").mkdir(parents=True)
    (project_root / "results" / "diagnostics").mkdir(parents=True)
    (project_root / "reports").mkdir(parents=True)
    (project_root / "diagnostics").mkdir(parents=True)
    (project_root / "inventory").mkdir(parents=True)
    (project_root / "inventory" / "inventory_paths.yaml").write_text(
        "paygap_root: ..\n", encoding="utf-8"
    )
    (project_root / "configs").mkdir(parents=True)

    variance_columns = ["stratifier", "stratum", "metric", "value", "n_obs"]
    pd.DataFrame(
        [
            {
                "model": "M6_reproductive",
                "female_coef": -0.12,
                "female_se": 0.01,
                "female_pvalue": 0.01,
                "r_squared": 0.3,
                "n_obs": 100,
                "controls": "x",
            }
        ]
    ).to_csv(project_root / "results" / "repro" / "acs_gap_ladder_extended.csv", index=False)
    pd.DataFrame(
        [
            {
                "year": 2023,
                "model": "M6_reproductive",
                "female_coef": -0.12,
                "female_se": 0.01,
                "female_pvalue": 0.01,
                "r_squared": 0.3,
                "n_obs": 100,
                "controls": "x",
            }
        ]
    ).to_csv(project_root / "results" / "repro" / "acs_gap_ladder_by_year.csv", index=False)
    pd.DataFrame(
        [
            {
                "sample": "childless_25_44",
                "outcome": "log_hourly_wage_real",
                "model": "fertility_risk",
                "term": "female",
                "coef": -0.1,
                "se": 0.02,
                "pvalue": 0.01,
                "n_obs": 40,
                "r_squared": 0.2,
            }
        ]
    ).to_csv(project_root / "results" / "repro" / "acs_fertility_risk_penalty.csv", index=False)
    pd.DataFrame(
        [
            {
                "sample": "childless_25_44",
                "outcome": "hourly_wage_real",
                "risk_quartile": "Q1",
                "mean_outcome": 20.0,
                "n_obs": 10,
                "weighted_n": 12.0,
            }
        ]
    ).to_csv(
        project_root / "results" / "repro" / "acs_fertility_risk_by_quartile.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "sample": "main_childless_25_44",
                "same_sex_couple_household": 0,
                "n_obs": 20,
                "mean_hourly_wage": 21.0,
                "mean_hours": 40.0,
                "mean_recent_birth": 0.1,
            }
        ]
    ).to_csv(project_root / "results" / "repro" / "acs_same_sex_placebos.csv", index=False)
    pd.DataFrame(
        [
            {
                "stratifier": "overall",
                "stratum": "all",
                "metric": "raw_variance_ratio",
                "value": 1.2,
                "n_obs": 100,
            },
            {
                "stratifier": "reproductive_stage",
                "stratum": "mother_under6",
                "metric": "male_top10_share",
                "value": 0.11,
                "n_obs": 50,
            },
            {
                "stratifier": "job_rigidity_quartile",
                "stratum": "Q4",
                "metric": "female_top10_share",
                "value": 0.08,
                "n_obs": 50,
            },
        ],
        columns=variance_columns,
    ).to_csv(project_root / "results" / "repro" / "acs_variance_suite.csv", index=False)
    pd.DataFrame(
        [
            {
                "stratifier": "overall",
                "stratum": "all",
                "metric": "male_top10_share",
                "value": 0.1,
                "n_obs": 100,
            }
        ],
        columns=variance_columns,
    ).to_csv(project_root / "results" / "repro" / "acs_tail_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "ok",
                "reproductive_stage": "overall",
                "metric": "minutes_paid_work_diary",
                "male_mean_minutes": 400,
                "female_mean_minutes": 360,
                "gap_minutes": -40,
                "n_male": 10,
                "n_female": 10,
                "weighted_n_male": 10.0,
                "weighted_n_female": 10.0,
            }
        ]
    ).to_csv(project_root / "results" / "repro" / "atus_mechanisms.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "ok_partial",
                "section": "status",
                "metric": "sample_size",
                "value": 100.0,
                "note": "stub",
            }
        ]
    ).to_csv(project_root / "results" / "repro" / "sipp_robustness.csv", index=False)

    schema = {
        "files": {
            "results/variance/acs_gap_ladder_extended.csv": [
                "model",
                "female_coef",
                "female_se",
                "female_pvalue",
                "r_squared",
                "n_obs",
                "controls",
            ],
            "results/variance/acs_gap_ladder_by_year.csv": [
                "year",
                "model",
                "female_coef",
                "female_se",
                "female_pvalue",
                "r_squared",
                "n_obs",
                "controls",
            ],
            "results/variance/acs_fertility_risk_penalty.csv": [
                "sample",
                "outcome",
                "model",
                "term",
                "coef",
                "se",
                "pvalue",
                "n_obs",
                "r_squared",
            ],
            "results/variance/acs_fertility_risk_by_quartile.csv": [
                "sample",
                "outcome",
                "risk_quartile",
                "mean_outcome",
                "n_obs",
                "weighted_n",
            ],
            "results/variance/acs_same_sex_placebos.csv": [
                "sample",
                "same_sex_couple_household",
                "n_obs",
                "mean_hourly_wage",
                "mean_hours",
                "mean_recent_birth",
            ],
            "results/variance/acs_variance_suite.csv": [
                "suite",
                "outcome",
                "stratifier",
                "stratum",
                "metric",
                "value",
                "n_obs",
            ],
            "results/variance/acs_tail_metrics.csv": [
                "suite",
                "outcome",
                "stratifier",
                "stratum",
                "metric",
                "value",
                "n_obs",
            ],
            "results/variance/acs_reproductive_dispersion.csv": [
                "suite",
                "outcome",
                "stratifier",
                "stratum",
                "metric",
                "value",
                "n_obs",
            ],
            "results/variance/acs_onet_dispersion.csv": [
                "suite",
                "outcome",
                "stratifier",
                "stratum",
                "metric",
                "value",
                "n_obs",
            ],
            "results/variance/acs_occupation_dispersion.csv": [
                "suite",
                "outcome",
                "stratifier",
                "stratum",
                "metric",
                "value",
                "n_obs",
                "occupation_harmonized_code",
                "occupation_harmonized_title",
                "occupation_harmonization_type",
                "occupation_title_vintage",
                "soc_major_group",
                "soc_major_label",
            ],
            "results/variance/acs_occupation_variability_leaders.csv": [
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
            ],
            "results/variance/acs_selection_corrected_variance.csv": [
                "status",
                "reason",
                "suite",
                "outcome",
                "stratifier",
                "stratum",
                "metric",
                "value",
                "n_obs",
            ],
            "results/diagnostics/variance_occupation_harmonization_map.csv": [
                "occupation_code_raw",
                "occupation_title_raw",
                "occupation_title_vintage",
                "occupation_mapping_regime",
                "occupation_harmonized_code",
                "occupation_harmonized_title",
                "occupation_harmonization_type",
                "soc_major_group",
                "soc_major_label",
            ],
            "results/variance/acs_tail_contrast_summary.csv": [
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
            ],
            "results/variance/acs_year_regime_variance_summary.csv": [
                "suite",
                "outcome",
                "metric",
                "regime",
                "weighted_mean_value",
                "n_obs_total",
            ],
            "results/variance/acs_soc_group_leaderboard_counts.csv": [
                "outcome",
                "leaderboard",
                "soc_major_group",
                "soc_major_label",
                "top10_count",
            ],
            "results/variance/acs_fertility_risk_variance_bridge.csv": [
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
            ],
            "results/variance/atus_mechanisms.csv": [
                "status",
                "reproductive_stage",
                "metric",
                "male_mean_minutes",
                "female_mean_minutes",
                "gap_minutes",
                "n_male",
                "n_female",
                "weighted_n_male",
                "weighted_n_female",
            ],
            "results/variance/sipp_robustness.csv": [
                "status",
                "section",
                "metric",
                "value",
                "note",
            ],
        }
    }
    (project_root / "configs" / "variance_output_schemas.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )

    monkeypatch.setattr(variance, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        variance,
        "load_variance_config",
        lambda path=None: {
            "datasets": {"acs": {"years": [2023]}, "onet": {"required_files": []}},
            "paths": {
                "results_dir": "results/variance",
                "report_path": "reports/variance_addon_summary.md",
                "inventory_usage_output": "results/diagnostics/variance_inventory_usage.csv",
                "inventory_usage_report": "reports/variance_inventory_usage.md",
                "optional_validation_output": (
                    "results/diagnostics/variance_optional_validation_status.csv"
                ),
                "inventory_config": "inventory/inventory_paths.yaml",
                "local_inventory_output": "diagnostics/variance_local_inventory_report.json",
                "atus_report_output": "reports/atus_variance_mechanisms.md",
                "onet_merge_coverage": "results/diagnostics/variance_onet_merge_coverage.csv",
                "occupation_dispersion_output": "results/variance/acs_occupation_dispersion.csv",
                "occupation_variability_leaders_output": (
                    "results/variance/acs_occupation_variability_leaders.csv"
                ),
                "occupation_harmonization_output": (
                    "results/diagnostics/variance_occupation_harmonization_map.csv"
                ),
                "release_manifest_output": "diagnostics/variance_release_manifest.json",
                "schema_snapshot_path": "configs/variance_output_schemas.json",
                "schema_check_output": "diagnostics/variance_schema_check.json",
            },
            "analysis": {
                "reproductive_stratifiers": [
                    "reproductive_stage",
                    "fertility_risk_quartile",
                    "couple_type",
                ],
                "onet_stratifiers": [
                    "job_rigidity_quartile",
                    "autonomy_quartile",
                    "time_pressure_quartile",
                ],
                "occupation_stratifier": "occupation_code",
                "occupation_min_n": 500,
                "occupation_top_k": 25,
            },
        },
    )
    monkeypatch.setattr(
        variance,
        "build_repro_inventory_usage",
        lambda years, files: pd.DataFrame(
            [
                {
                    "asset_group": "acs",
                    "asset_name": "acs_pums_2023_api_repweights.parquet",
                    "status": "present",
                    "legacy_path": "data/raw/acs/acs_pums_2023_api_repweights.parquet",
                    "canonical_path": (
                        "sources/census/acs/wave2/paygap/raw/acs/"
                        "acs_pums_2023_api_repweights.parquet"
                    ),
                    "note": "variance_addon",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        variance,
        "build_optional_validation_status",
        lambda: pd.DataFrame(
            [
                {
                    "dataset": "NLSY79",
                    "status": "ready",
                    "canonical_path": "",
                    "expected_processed": "",
                    "note": "stub",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        variance,
        "build_local_inventory_report",
        lambda path: {
            "inventory_path": str(path),
            "generated_at": "2026-03-16T00:00:00+00:00",
            "summary": {"configured_paths": 1, "existing_paths": 1, "missing_paths": 0},
            "checks": {
                "paygap_root": {
                    "configured": True,
                    "exists": True,
                    "path": str(project_root),
                    "missing_children": [],
                }
            },
        },
    )
    monkeypatch.setattr(
        variance,
        "_write_atus_outputs",
        lambda results_dir, report_path: (
            "ok: reused repro ATUS table",
            (results_dir / "atus_mechanisms.csv"),
        ),
    )
    monkeypatch.setattr(
        variance,
        "_write_sipp_outputs",
        lambda results_dir: ("ok: reused repro SIPP table", (results_dir / "sipp_robustness.csv")),
    )
    monkeypatch.setattr(variance, "write_nlsy_validation_output", lambda output_path: None)

    outputs = variance.run_variance_addon()

    variance_suite = pd.read_csv(project_root / "results" / "variance" / "acs_variance_suite.csv")
    selection = pd.read_csv(
        project_root / "results" / "variance" / "acs_selection_corrected_variance.csv"
    )

    assert outputs["summary_report"] == project_root / "reports" / "variance_addon_summary.md"
    assert "suite" in variance_suite.columns
    assert "outcome" in variance_suite.columns
    assert set(
        pd.read_csv(project_root / "results" / "variance" / "acs_reproductive_dispersion.csv")[
            "stratifier"
        ]
    ) == {"reproductive_stage"}
    assert set(
        pd.read_csv(project_root / "results" / "variance" / "acs_onet_dispersion.csv")[
            "stratifier"
        ]
    ) == {"job_rigidity_quartile"}
    assert selection["status"].iloc[0] == "skipped"
    assert (
        project_root / "results" / "diagnostics" / "variance_occupation_harmonization_map.csv"
    ).exists()
    assert (project_root / "results" / "variance" / "acs_tail_contrast_summary.csv").exists()
    assert (
        project_root / "results" / "variance" / "acs_year_regime_variance_summary.csv"
    ).exists()
    assert (
        project_root / "results" / "variance" / "acs_soc_group_leaderboard_counts.csv"
    ).exists()
    assert (
        project_root / "results" / "variance" / "acs_fertility_risk_variance_bridge.csv"
    ).exists()
    assert (
        json.loads((project_root / "diagnostics" / "variance_release_manifest.json").read_text())[
            "project"
        ]
        == "paygap_variance_addon"
    )


def test_selection_corrected_variance_derives_log_wage_from_hourly(tmp_path):
    rng = np.random.default_rng(0)
    n = 240
    female = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    age = rng.integers(25, 55, n)
    panel = pd.DataFrame(
        {
            "female": female,
            "age": age,
            "age_sq": age**2,
            "race_ethnicity": ["white_non_hispanic"] * n,
            "education_level": ["bachelors"] * n,
            "marital_status": ["married"] * n,
            "number_children": rng.choice([0, 1, 2], size=n),
            "children_under_5": rng.binomial(1, 0.2, size=n),
            "state_fips": [6] * n,
            "person_weight": rng.uniform(1.0, 3.0, size=n),
            "employment_indicator": rng.binomial(1, 0.82, size=n),
            "hourly_wage_real": rng.uniform(15.0, 55.0, size=n),
            "survey_year": [2023] * n,
        }
    )
    panel.loc[panel["employment_indicator"] == 0, "hourly_wage_real"] = np.nan

    path = variance._write_selection_corrected_variance(panel, tmp_path)
    result = pd.read_csv(path)

    assert result["status"].iloc[0] == "ok"
    assert result["value"].notna().any()


def test_write_panel_derived_outputs_emits_hourly_and_annual_variance(monkeypatch, tmp_path):
    n = 24
    panel = pd.DataFrame(
        {
            "female": ([0, 1] * (n // 2)),
            "person_weight": np.linspace(1.0, 2.0, n),
            "survey_year": ([2023] * (n // 2)) + ([2024] * (n // 2)),
            "hourly_wage_real": np.linspace(20.0, 43.0, n),
            "annual_earnings_real": np.linspace(40000.0, 86000.0, n),
            "occupation_code": ["1010"] * (n // 2) + ["2010"] * (n // 2),
            "reproductive_stage": ["childless"] * n,
            "fertility_risk_quartile": ["Q1"] * n,
            "couple_type": ["different_sex"] * n,
            "job_rigidity_quartile": ["Q2"] * n,
            "autonomy_quartile": ["Q3"] * n,
            "time_pressure_quartile": ["Q4"] * n,
        }
    )
    panel = variance._ensure_logged_outcomes(panel)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "diagnostics").mkdir()
    outputs = {}
    addon_outputs = []

    monkeypatch.setattr(
        variance,
        "run_sequential_ols",
        lambda *args, **kwargs: {"M6_reproductive": {}},
    )
    monkeypatch.setattr(
        variance,
        "results_to_dataframe",
        lambda _: pd.DataFrame(
            [
                {
                    "model": "M6_reproductive",
                    "female_coef": -0.1,
                    "female_se": 0.01,
                    "female_pvalue": 0.05,
                    "r_squared": 0.2,
                    "n_obs": n,
                    "controls": "stub",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        variance,
        "coefficient_table",
        lambda *args, **kwargs: pd.DataFrame(
            [{"term": "female_x_job_rigidity", "coef": -0.01, "se": 0.01, "pvalue": 0.2}]
        ),
    )
    monkeypatch.setattr(
        variance,
        "run_fertility_risk_penalty",
        lambda *_: (
            pd.DataFrame(
                [
                    {
                        "sample": "childless_25_44",
                        "outcome": "log_hourly_wage_real",
                        "model": "fertility_risk",
                        "term": "female",
                        "coef": -0.1,
                        "se": 0.02,
                        "pvalue": 0.01,
                        "n_obs": n,
                        "r_squared": 0.2,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "sample": "childless_25_44",
                        "outcome": "log_hourly_wage_real",
                        "risk_quartile": "Q1",
                        "mean_outcome": 20.0,
                        "n_obs": n,
                        "weighted_n": float(n),
                    }
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        variance,
        "build_same_sex_placebos",
        lambda *_: pd.DataFrame(
            [
                {
                    "sample": "main_childless_25_44",
                    "same_sex_couple_household": 0,
                    "n_obs": n,
                    "mean_hourly_wage": 21.0,
                    "mean_hours": 40.0,
                    "mean_recent_birth": 0.1,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        variance,
        "attach_occupation_metadata",
        lambda frame, code_col, survey_year_col: frame.assign(
            occupation_code_raw=frame[code_col].astype("string"),
            occupation_title_raw=np.where(
                frame[code_col].astype("string") == "1010",
                "Software developers",
                "HR specialists",
            ),
            occupation_title_vintage=np.where(
                pd.to_numeric(frame[survey_year_col], errors="coerce") <= 2018,
                "2010",
                "2018",
            ),
            occupation_harmonized_code=frame[code_col].astype("string"),
            occupation_harmonized_title=np.where(
                frame[code_col].astype("string") == "1010",
                "Software developers",
                "HR specialists",
            ),
            occupation_harmonization_type="native_2018",
            soc_major_group=np.where(
                frame[code_col].astype("string") == "1010",
                "15",
                "13",
            ),
            soc_major_label=np.where(
                frame[code_col].astype("string") == "1010",
                "Computer and Mathematical",
                "Business and Financial Operations",
            ),
        ),
    )

    variance._write_panel_derived_outputs(
        panel=panel,
        results_dir=results_dir,
        analysis={
            "outcomes": ["log_hourly_wage_real", "log_annual_earnings_real"],
            "reproductive_stratifiers": [
                "reproductive_stage",
                "fertility_risk_quartile",
                "couple_type",
            ],
            "onet_stratifiers": [
                "job_rigidity_quartile",
                "autonomy_quartile",
                "time_pressure_quartile",
            ],
        },
        outputs=outputs,
        addon_outputs=addon_outputs,
        occupation_harmonization_output=results_dir / "diagnostics" / "variance_occ_map.csv",
    )

    variance_suite = pd.read_csv(results_dir / "acs_variance_suite.csv")
    occupation_dispersion = pd.read_csv(results_dir / "acs_occupation_dispersion.csv")
    occupation_leaders = pd.read_csv(results_dir / "acs_occupation_variability_leaders.csv")
    assert set(variance_suite["outcome"]) == {
        "log_hourly_wage_real",
        "log_annual_earnings_real",
    }
    assert {
        "occupation_harmonized_code",
        "occupation_harmonized_title",
        "soc_major_label",
    }.issubset(occupation_dispersion.columns)
    assert {
        "occupation_harmonized_code",
        "occupation_harmonized_title",
        "soc_major_label",
    }.issubset(occupation_leaders.columns)


def test_attach_occupation_metadata_year_aware_passes_survey_year(monkeypatch):
    captured = {}

    def _fake_attach(frame, code_col, survey_year_col):
        captured["code_col"] = code_col
        captured["survey_year_col"] = survey_year_col
        return frame.assign(
            occupation_code_raw=frame[code_col].astype("string"),
            occupation_title_raw="Software developers",
            occupation_title_vintage="2018",
            occupation_harmonized_code=frame[code_col].astype("string"),
            occupation_harmonized_title="Software developers",
            occupation_harmonization_type="native_2018",
            soc_major_group="15",
            soc_major_label="Computer and Mathematical",
        )

    monkeypatch.setattr(variance, "attach_occupation_metadata", _fake_attach)
    attached = variance._attach_occupation_metadata_year_aware(
        pd.DataFrame([{"stratum": "1010", "survey_year": 2023, "n_obs": 10}]),
        code_col="stratum",
        survey_year_col="survey_year",
        require_year_context=True,
    )

    assert captured["code_col"] == "stratum"
    assert captured["survey_year_col"] == "survey_year"
    assert attached.loc[0, "occupation_harmonized_code"] == "1010"
