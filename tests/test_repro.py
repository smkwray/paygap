"""Tests for reproductive extension path resolution."""

from pathlib import Path

import pandas as pd

from gender_gap import repro


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


def test_acs_raw_columns_handle_2019_plus_schema():
    cols = repro._acs_raw_columns(2023)

    assert "RELSHIPP" in cols
    assert "CPLT" in cols
    assert "RELP" not in cols


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
