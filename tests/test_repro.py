"""Tests for reproductive extension path resolution."""

from pathlib import Path

import pandas as pd
import pytest

from gender_gap import repro
from gender_gap.models.ols import OLSResult


def test_resolve_acs_raw_path_prefers_shared_when_local_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "PROJECT_ROOT", tmp_path)
    shared_file = tmp_path / "shared" / "sources" / "census" / "acs" / "wave2" / "paygap" / "raw" / "acs" / "acs_pums_2013_api_repweights.parquet"
    shared_file.parent.mkdir(parents=True, exist_ok=True)
    shared_file.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(repro, "shared_source_path", lambda *parts: tmp_path / "shared" / "sources" / Path(*parts))

    resolved = repro._resolve_acs_raw_path(2013)

    assert resolved == shared_file


def test_resolve_acs_raw_path_prefers_shared_when_both_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "PROJECT_ROOT", tmp_path)
    local_file = tmp_path / "data" / "raw" / "acs" / "acs_pums_2013_api_repweights.parquet"
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text("local", encoding="utf-8")
    shared_file = tmp_path / "shared" / "sources" / "census" / "acs" / "wave2" / "paygap" / "raw" / "acs" / "acs_pums_2013_api_repweights.parquet"
    shared_file.parent.mkdir(parents=True, exist_ok=True)
    shared_file.write_text("shared", encoding="utf-8")
    monkeypatch.setattr(repro, "shared_source_path", lambda *parts: tmp_path / "shared" / "sources" / Path(*parts))

    resolved = repro._resolve_acs_raw_path(2013)

    assert resolved == shared_file


def test_resolve_onet_dir_prefers_shared_when_local_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "PROJECT_ROOT", tmp_path)
    shared_dir = tmp_path / "shared" / "sources" / "onet" / "db_30_2_text"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "Work Context.txt").write_text("stub", encoding="utf-8")
    (shared_dir / "Scales Reference.txt").write_text("stub", encoding="utf-8")
    monkeypatch.setattr(repro, "shared_source_path", lambda *parts: tmp_path / "shared" / "sources" / Path(*parts))

    resolved = repro._resolve_onet_dir(["Work Context.txt", "Scales Reference.txt"])

    assert resolved == shared_dir


def test_resolve_onet_dir_prefers_shared_when_both_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "PROJECT_ROOT", tmp_path)
    local_dir = tmp_path / "data" / "raw" / "context" / "onet" / "db_30_2_text"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "Work Context.txt").write_text("local", encoding="utf-8")
    (local_dir / "Scales Reference.txt").write_text("local", encoding="utf-8")
    shared_dir = tmp_path / "shared" / "sources" / "onet" / "db_30_2_text"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "Work Context.txt").write_text("shared", encoding="utf-8")
    (shared_dir / "Scales Reference.txt").write_text("shared", encoding="utf-8")
    monkeypatch.setattr(repro, "shared_source_path", lambda *parts: tmp_path / "shared" / "sources" / Path(*parts))

    resolved = repro._resolve_onet_dir(["Work Context.txt", "Scales Reference.txt"])

    assert resolved == shared_dir


def test_acs_raw_columns_handle_pre_2019_schema():
    cols = repro._acs_raw_columns(2018)

    assert "RELP" in cols
    assert "RELSHIPP" not in cols
    assert "CPLT" not in cols
    assert "MULTG" in cols


def test_acs_raw_columns_handle_2019_plus_schema():
    cols = repro._acs_raw_columns(2023)

    assert "RELSHIPP" in cols
    assert "CPLT" in cols
    assert "RELP" not in cols
    assert "MULTG" in cols


def test_acs_raw_columns_include_state_fallback_for_2023plus_api_extracts():
    cols = repro._acs_raw_columns(2024)

    assert "ST" in cols
    assert "STATE" in cols


def test_available_columns_ignores_missing_parquet_fields(tmp_path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"SERIALNO": ["1"], "SPORDER": [1], "PWGTP": [10]}).to_parquet(path, index=False)

    cols = repro._available_columns(path, ["SERIALNO", "POWSP", "PWGTP"])

    assert cols == ["SERIALNO", "PWGTP"]


def test_compress_panel_keeps_requested_columns_and_downcasts():
    df = pd.DataFrame(
        {
            "survey_year": [2023, 2023],
            "female": [1, 0],
            "race_ethnicity": ["white_non_hispanic", "black"],
            "hourly_wage_real": [25.0, 30.0],
            "drop_me": ["x", "y"],
        }
    )

    compressed = repro._compress_panel(df, {"survey_year", "female", "race_ethnicity", "hourly_wage_real"})

    assert list(compressed.columns) == ["survey_year", "female", "race_ethnicity", "hourly_wage_real"]
    assert str(compressed["race_ethnicity"].dtype) == "category"


def test_compress_panel_coerces_object_numeric_columns():
    df = pd.DataFrame(
        {
            "survey_year": [2023, 2023],
            "female": [1, 0],
            "hourly_wage_real": ["25.0", "30.0"],
            "log_hourly_wage_real": ["3.2189", "3.4012"],
        }
    )

    compressed = repro._compress_panel(df, {"survey_year", "female", "hourly_wage_real", "log_hourly_wage_real"})

    assert pd.api.types.is_numeric_dtype(compressed["hourly_wage_real"])
    assert pd.api.types.is_numeric_dtype(compressed["log_hourly_wage_real"])


def test_apply_analysis_sample_prime_age_wage_salary():
    df = pd.DataFrame(
        {
            "age": [30, 24, 40],
            "self_employed": [0, 0, 1],
            "usual_hours_week": [40, 40, 40],
            "hourly_wage_real": [20.0, 20.0, 20.0],
        }
    )

    filtered = repro._apply_analysis_sample(df, "prime_age_wage_salary")

    assert len(filtered) == 1
    assert filtered["age"].iloc[0] == 30


def test_load_acs_year_applies_household_enrichment(tmp_path, monkeypatch):
    raw_path = tmp_path / "acs.parquet"
    raw_path.write_text("stub", encoding="utf-8")

    standardized = pd.DataFrame(
        {
            "acs_serialno": ["HH1", "HH1", "HH2"],
            "acs_sporder": [1, 2, 1],
            "relshipp": [20, 21, 20],
            "age": [35, 33, 29],
            "wage_salary_income_real": [60000.0, 45000.0, 52000.0],
            "annual_earnings_real": [65000.0, 48000.0, 54000.0],
            "multg": [2, 2, 1],
            "female": [0, 1, 1],
            "number_children": [1, 1, 0],
            "marital_status": ["married", "married", "never_married"],
            "same_sex_couple_household": [0, 0, 0],
            "opposite_sex_couple_household": [1, 1, 0],
        }
    )

    monkeypatch.setattr(repro, "_resolve_acs_raw_path", lambda year: raw_path)
    monkeypatch.setattr(repro, "_available_columns", lambda path, requested: requested)
    monkeypatch.setattr(repro, "read_parquet", lambda path, columns=None: pd.DataFrame({"SERIALNO": ["stub"]}))
    monkeypatch.setattr(repro, "standardize_acs", lambda raw_df, survey_year, keep_replicate_weights: standardized.copy())

    loaded = repro._load_acs_year(2023)

    assert loaded.loc[loaded["acs_sporder"] == 1, "multigenerational"].iloc[0] == pytest.approx(1.0)
    assert loaded.loc[loaded["acs_sporder"] == 1, "other_adults_present"].iloc[0] == 0
    assert loaded.loc[loaded["acs_sporder"] == 1, "partner_wage_real"].iloc[0] == pytest.approx(45000.0)


def test_build_household_sensitivity_emits_expected_panels(monkeypatch):
    panel = pd.DataFrame(
        {
            "hourly_wage_real": [22.2, 20.1, 24.5, 27.1],
            "person_weight": [1.0, 1.2, 0.9, 1.1],
            "female": [1, 0, 1, 0],
            "age": [30, 31, 32, 33],
            "age_sq": [900, 961, 1024, 1089],
            "race_ethnicity": ["white", "white", "black", "black"],
            "education_level": ["ba", "ba", "ma", "ma"],
            "state_fips": [36, 36, 6, 6],
            "occupation_code": [100, 100, 200, 200],
            "industry_code": [10, 10, 20, 20],
            "class_of_worker": [1, 1, 1, 1],
            "usual_hours_week": [40, 42, 38, 40],
            "work_from_home": [0, 0, 1, 1],
            "commute_minutes_one_way": [30, 35, 20, 25],
            "marital_status": ["married", "married", "married", "married"],
            "number_children": [1, 1, 0, 0],
            "children_under_5": [1, 1, 0, 0],
            "recent_birth": [0, 0, 0, 0],
            "recent_marriage": [0, 0, 0, 0],
            "has_own_child": [1, 1, 0, 0],
            "own_child_under6": [1, 1, 0, 0],
            "own_child_6_17_only": [0, 0, 0, 0],
            "couple_type": ["opposite_sex", "opposite_sex", "opposite_sex", "opposite_sex"],
            "reproductive_stage": ["mother_under6", "mother_under6", "childless_other_partnered", "childless_other_partnered"],
            "autonomy": [0.5, 0.5, 0.7, 0.7],
            "schedule_unpredictability": [0.2, 0.2, 0.4, 0.4],
            "time_pressure": [0.4, 0.4, 0.3, 0.3],
            "coordination_responsibility": [0.3, 0.3, 0.5, 0.5],
            "physical_proximity": [0.2, 0.2, 0.4, 0.4],
            "job_rigidity": [0.6, 0.6, 0.5, 0.5],
            "multigenerational": [1, 1, 0, 0],
            "other_adults_present": [0, 0, 1, 1],
            "partner_employed": [1.0, 1.0, 1.0, 1.0],
            "partner_wage_real": [45000.0, 60000.0, 70000.0, 68000.0],
            "relative_earnings": [0.4, 0.6, 0.3, 0.7],
        }
    )

    calls: list[dict] = []

    def fake_run_sequential_ols(df, outcome="log_hourly_wage_real", weight_col="person_weight", blocks=None):
        calls.append({"columns": list(df.columns), "blocks": blocks})
        return [
            OLSResult(
                model_name=model_name,
                female_coef=-0.1,
                female_se=0.01,
                female_pvalue=0.01,
                r_squared=0.4,
                n_obs=len(df),
                controls=controls,
            )
            for model_name, controls in blocks.items()
        ]

    monkeypatch.setattr(repro, "run_sequential_ols", fake_run_sequential_ols)

    sensitivity = repro._build_household_sensitivity(panel)

    assert set(sensitivity["panel"]) == {"household_composition", "partner_resources"}
    assert set(sensitivity["sample"]) == {"full_sample", "partnered_households"}
    assert set(sensitivity["model"]) == {
        "M7_onet_context",
        "M7_onet_context_plus_household_composition",
        "M7_onet_context_partnered_baseline",
        "M7_onet_context_plus_partner_resources",
    }
    assert len(calls) == 2
    assert all("hourly_wage_real" in call["columns"] for call in calls)
    assert all("relative_earnings" not in term for call in calls for controls in call["blocks"].values() for term in controls)
    composition_blocks = calls[0]["blocks"]
    assert composition_blocks["M7_onet_context_plus_household_composition"][-2:] == [
        "multigenerational",
        "other_adults_present",
    ]
    resource_blocks = calls[1]["blocks"]
    assert resource_blocks["M7_onet_context_plus_partner_resources"][-2:] == [
        "partner_employed",
        "partner_wage_real",
    ]


def test_build_household_sensitivity_drops_all_missing_household_terms(monkeypatch):
    panel = pd.DataFrame(
        {
            "hourly_wage_real": [22.2, 20.1, 24.5, 27.1],
            "person_weight": [1.0, 1.2, 0.9, 1.1],
            "female": [1, 0, 1, 0],
            "age": [30, 31, 32, 33],
            "age_sq": [900, 961, 1024, 1089],
            "race_ethnicity": ["white", "white", "black", "black"],
            "education_level": ["ba", "ba", "ma", "ma"],
            "state_fips": [36, 36, 6, 6],
            "occupation_code": [100, 100, 200, 200],
            "industry_code": [10, 10, 20, 20],
            "class_of_worker": [1, 1, 1, 1],
            "usual_hours_week": [40, 42, 38, 40],
            "work_from_home": [0, 0, 1, 1],
            "commute_minutes_one_way": [30, 35, 20, 25],
            "marital_status": ["married", "married", "married", "married"],
            "number_children": [1, 1, 0, 0],
            "children_under_5": [1, 1, 0, 0],
            "recent_birth": [0, 0, 0, 0],
            "recent_marriage": [0, 0, 0, 0],
            "has_own_child": [1, 1, 0, 0],
            "own_child_under6": [1, 1, 0, 0],
            "own_child_6_17_only": [0, 0, 0, 0],
            "couple_type": ["opposite_sex", "opposite_sex", "opposite_sex", "opposite_sex"],
            "reproductive_stage": [
                "mother_under6",
                "mother_under6",
                "childless_other_partnered",
                "childless_other_partnered",
            ],
            "autonomy": [0.5, 0.5, 0.7, 0.7],
            "schedule_unpredictability": [0.2, 0.2, 0.4, 0.4],
            "time_pressure": [0.4, 0.4, 0.3, 0.3],
            "coordination_responsibility": [0.3, 0.3, 0.5, 0.5],
            "physical_proximity": [0.2, 0.2, 0.4, 0.4],
            "job_rigidity": [0.6, 0.6, 0.5, 0.5],
            "multigenerational": [float("nan")] * 4,
            "other_adults_present": [0, 0, 1, 1],
            "partner_employed": [1.0, 1.0, 1.0, 1.0],
            "partner_wage_real": [45000.0, 60000.0, 70000.0, 68000.0],
        }
    )

    captured_blocks: list[dict[str, list[str]]] = []

    def fake_run_sequential_ols(df, outcome="log_hourly_wage_real", weight_col="person_weight", blocks=None):
        captured_blocks.append(blocks)
        return [
            OLSResult(
                model_name=model_name,
                female_coef=-0.1,
                female_se=0.01,
                female_pvalue=0.01,
                r_squared=0.4,
                n_obs=len(df),
                controls=controls,
            )
            for model_name, controls in blocks.items()
        ]

    monkeypatch.setattr(repro, "run_sequential_ols", fake_run_sequential_ols)

    repro._build_household_sensitivity(panel)

    composition_controls = captured_blocks[0]["M7_onet_context_plus_household_composition"]
    assert "multigenerational" not in composition_controls
    assert "other_adults_present" in composition_controls
