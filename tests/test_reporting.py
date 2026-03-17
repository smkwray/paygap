"""Tests for reporting modules: charts, artifacts, tables."""

import json
from pathlib import Path

import pandas as pd
import pytest

from gender_gap.models.ols import OLSResult
from gender_gap.reporting.artifacts import export_json_artifacts
from gender_gap.reporting.charts import (
    generate_all_plots,
    plot_heterogeneity_forest,
    plot_oaxaca_decomposition,
    plot_ols_sequential,
    plot_quantile_coefficients,
)
from gender_gap.reporting.tables import (
    export_markdown_summary,
)
from gender_gap.reporting import repro as repro_reporting


@pytest.fixture
def ols_df():
    return pd.DataFrame({
        "model": ["M0", "M1", "M2"],
        "female_coef": [-0.25, -0.18, -0.12],
        "female_se": [0.01, 0.01, 0.015],
        "female_pvalue": [0.0001, 0.0001, 0.0001],
        "r_squared": [0.05, 0.15, 0.30],
        "n_obs": [10000, 10000, 10000],
        "controls": ["none", "age", "age+ed"],
    })


@pytest.fixture
def quantile_df():
    return pd.DataFrame({
        "quantile": [0.10, 0.25, 0.50, 0.75, 0.90],
        "female_coef": [-0.08, -0.12, -0.15, -0.18, -0.22],
        "female_se": [0.02, 0.015, 0.01, 0.012, 0.025],
        "female_pvalue": [0.001] * 5,
        "n_obs": [5000] * 5,
    })


@pytest.fixture
def het_df():
    return pd.DataFrame({
        "group": ["hs", "ba", "grad"],
        "gap": [-0.05, -0.12, -0.20],
        "se": [0.02, 0.015, 0.025],
        "ci_lower": [-0.09, -0.15, -0.25],
        "ci_upper": [-0.01, -0.09, -0.15],
        "n": [3000, 4000, 2000],
    })


@pytest.fixture
def oaxaca_df():
    return pd.DataFrame({
        "explained": [0.08],
        "unexplained": [0.12],
        "total_gap": [0.20],
    })


@pytest.fixture
def model_output_dir(tmp_path, ols_df, quantile_df, het_df, oaxaca_df):
    """Create a tmp directory with model output CSVs."""
    ols_df.to_csv(tmp_path / "ols_sequential.csv", index=False)
    quantile_df.to_csv(tmp_path / "quantile_regression.csv", index=False)
    het_df.to_csv(tmp_path / "heterogeneity_education_level.csv", index=False)
    oaxaca_df.to_csv(tmp_path / "oaxaca.csv", index=False)
    pd.DataFrame([{
        "male_mean": 30.0, "female_mean": 25.0,
        "gap_dollars": 5.0, "gap_pct": 16.7,
        "n_male": 5000, "n_female": 5000,
    }]).to_csv(tmp_path / "raw_gap.csv", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "models_run": ["ols", "quantile"], "n_obs": 10000,
    }))
    return tmp_path


# --- Charts tests ---

class TestCharts:
    def test_plot_ols_sequential(self, tmp_path, ols_df):
        path = plot_ols_sequential(ols_df, tmp_path / "ols.png")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_plot_quantile_coefficients(self, tmp_path, quantile_df):
        path = plot_quantile_coefficients(quantile_df, tmp_path / "quantile.png")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_plot_heterogeneity_forest(self, tmp_path, het_df):
        path = plot_heterogeneity_forest(het_df, "education_level", tmp_path / "het.png")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_plot_oaxaca_decomposition(self, tmp_path, oaxaca_df):
        path = plot_oaxaca_decomposition(oaxaca_df, tmp_path / "oaxaca.png")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_generate_all_plots(self, model_output_dir, tmp_path):
        out = tmp_path / "plots"
        plots = generate_all_plots(model_output_dir, out)
        assert len(plots) >= 3  # ols, quantile, oaxaca at minimum
        for p in plots:
            assert p.exists()


# --- Artifacts tests ---

class TestArtifacts:
    def test_export_json_artifacts(self, model_output_dir, tmp_path):
        out = tmp_path / "artifacts"
        path = export_json_artifacts(model_output_dir, out)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "version" in data
        assert "models" in data
        assert "ols_sequential" in data["models"]

    def test_json_has_quantile(self, model_output_dir, tmp_path):
        out = tmp_path / "artifacts"
        path = export_json_artifacts(model_output_dir, out)
        data = json.loads(path.read_text())
        assert "quantile_regression" in data["models"]
        assert len(data["models"]["quantile_regression"]) == 5

    def test_json_has_heterogeneity(self, model_output_dir, tmp_path):
        out = tmp_path / "artifacts"
        path = export_json_artifacts(model_output_dir, out)
        data = json.loads(path.read_text())
        assert "heterogeneity" in data["models"]
        assert "education_level" in data["models"]["heterogeneity"]

    def test_json_has_manifest(self, model_output_dir, tmp_path):
        out = tmp_path / "artifacts"
        path = export_json_artifacts(model_output_dir, out)
        data = json.loads(path.read_text())
        assert "manifest" in data
        assert data["manifest"]["n_obs"] == 10000

    def test_empty_input_dir(self, tmp_path):
        inp = tmp_path / "empty"
        inp.mkdir()
        out = tmp_path / "out"
        path = export_json_artifacts(inp, out)
        data = json.loads(path.read_text())
        assert data["models"] == {}


class TestReproReporting:
    def test_build_optional_validation_status_detects_raw_nlsy_and_public_psid(self, tmp_path, monkeypatch):
        project_root = tmp_path / "paygap"
        project_root.mkdir(parents=True)
        (project_root / "data" / "external" / "nlsy").mkdir(parents=True)

        shared_root = tmp_path / "shared"
        catalog_dir = shared_root / "catalog"
        catalog_dir.mkdir(parents=True)
        (catalog_dir / "datasets.csv").write_text("canonical_path,source_url\n", encoding="utf-8")
        (catalog_dir / "aliases.csv").write_text("project,legacy_path,canonical_path,status\n", encoding="utf-8")

        nlsy_dir = shared_root / "sources" / "misc" / "large_payloads" / "wave3c" / "sexg" / "raw"
        nlsy_dir.mkdir(parents=True)
        (nlsy_dir / "nlsy79_all_1979-2022.zip").write_text("stub", encoding="utf-8")
        (nlsy_dir / "nlsy97_all_1997-2023.zip").write_text("stub", encoding="utf-8")

        psid_dir = shared_root / "sources" / "umich" / "psid_cds_tas" / "public"
        psid_dir.mkdir(parents=True)

        monkeypatch.setattr(repro_reporting, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(repro_reporting, "SHARED_DATASETS_CATALOG", catalog_dir / "datasets.csv")
        monkeypatch.setattr(repro_reporting, "SHARED_ALIASES_CATALOG", catalog_dir / "aliases.csv")
        monkeypatch.setattr(
            repro_reporting,
            "shared_source_path",
            lambda *parts: shared_root / "sources" / Path(*parts),
        )

        status = repro_reporting.build_optional_validation_status()

        assert status.set_index("dataset").loc["NLSY79", "status"] == "raw_only"
        assert status.set_index("dataset").loc["NLSY97", "status"] == "raw_only"
        assert status.set_index("dataset").loc["PSID", "status"] == "public_bundle_only"

    def test_build_optional_validation_status_detects_raw_main_panel_psid(self, tmp_path, monkeypatch):
        project_root = tmp_path / "paygap"
        project_root.mkdir(parents=True)
        (project_root / "data" / "external" / "nlsy").mkdir(parents=True)

        shared_root = tmp_path / "shared"
        catalog_dir = shared_root / "catalog"
        catalog_dir.mkdir(parents=True)
        (catalog_dir / "datasets.csv").write_text("canonical_path,source_url\n", encoding="utf-8")
        (catalog_dir / "aliases.csv").write_text("project,legacy_path,canonical_path,status\n", encoding="utf-8")

        psid_raw_dir = shared_root / "sources" / "umich" / "psid" / "main_public" / "paygap" / "raw" / "psid"
        psid_raw_dir.mkdir(parents=True)
        (psid_raw_dir / "psid_family_2023.zip").write_text("stub", encoding="utf-8")

        monkeypatch.setattr(repro_reporting, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(repro_reporting, "SHARED_DATASETS_CATALOG", catalog_dir / "datasets.csv")
        monkeypatch.setattr(repro_reporting, "SHARED_ALIASES_CATALOG", catalog_dir / "aliases.csv")
        monkeypatch.setattr(
            repro_reporting,
            "shared_source_path",
            lambda *parts: shared_root / "sources" / Path(*parts),
        )

        status = repro_reporting.build_optional_validation_status()

        assert status.set_index("dataset").loc["PSID", "status"] == "raw_main_panel"
        assert status.set_index("dataset").loc["PSID", "expected_processed"] == "data/external/psid/psid_2023_analysis_ready.parquet"

    def test_build_optional_validation_status_detects_ready_psid(self, tmp_path, monkeypatch):
        project_root = tmp_path / "paygap"
        project_root.mkdir(parents=True)
        (project_root / "data" / "external" / "nlsy").mkdir(parents=True)

        shared_root = tmp_path / "shared"
        catalog_dir = shared_root / "catalog"
        catalog_dir.mkdir(parents=True)
        (catalog_dir / "datasets.csv").write_text(
            "canonical_path,source_url\n"
            "data/sources/umich/psid/main_public/paygap/processed/psid/psid_2023_analysis_ready.parquet,test-source\n",
            encoding="utf-8",
        )
        (catalog_dir / "aliases.csv").write_text("project,legacy_path,canonical_path,status\n", encoding="utf-8")

        psid_processed_dir = shared_root / "sources" / "umich" / "psid" / "main_public" / "paygap" / "processed" / "psid"
        psid_processed_dir.mkdir(parents=True)
        (psid_processed_dir / "psid_2023_analysis_ready.parquet").write_text("stub", encoding="utf-8")

        monkeypatch.setattr(repro_reporting, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(repro_reporting, "SHARED_DATASETS_CATALOG", catalog_dir / "datasets.csv")
        monkeypatch.setattr(repro_reporting, "SHARED_ALIASES_CATALOG", catalog_dir / "aliases.csv")
        monkeypatch.setattr(
            repro_reporting,
            "shared_source_path",
            lambda *parts: shared_root / "sources" / Path(*parts),
        )

        status = repro_reporting.build_optional_validation_status()

        assert status.set_index("dataset").loc["PSID", "status"] == "ready"

    def test_write_repro_summary_includes_optional_validation_section(self, tmp_path):
        out = tmp_path / "summary.md"
        optional = pd.DataFrame(
            [
                {"dataset": "NLSY79", "status": "raw_only", "note": "Needs adapter."},
                {"dataset": "PSID", "status": "public_bundle_only", "note": "Public bundle only."},
            ]
        )

        repro_reporting.write_repro_summary(
            output_path=out,
            available_years=[2023],
            inventory_usage=pd.DataFrame([{"x": 1}]),
            missing_inputs=[],
            generated_files=[tmp_path / "results.csv"],
            optional_validation=optional,
        )

        text = out.read_text()
        assert "## Optional validation data status" in text
        assert "NLSY79: raw_only: Needs adapter." in text
        assert "PSID: public_bundle_only: Public bundle only." in text


# --- Tables tests ---

class TestTables:
    def test_export_markdown_summary(self, tmp_path):
        raw = {"male_mean": 30.0, "female_mean": 25.0,
               "gap_dollars": 5.0, "gap_pct": 16.7,
               "n_male": 5000, "n_female": 5000}
        ols_results = [
            OLSResult("M0", -0.25, 0.01, 0.0001, 0.05, 10000, "none"),
        ]
        path = export_markdown_summary(raw, ols_results, tmp_path)
        assert path.exists()
        text = path.read_text()
        assert "Gender Earnings Gap" in text
        assert "$30.00" in text
        assert "M0" in text
