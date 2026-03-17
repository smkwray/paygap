"""Tests for the CLI entry point."""

from pathlib import Path

import pandas as pd
import pytest

from gender_gap.cli import main


class TestCLIBasics:
    def test_no_args_shows_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "gender-gap" in captured.out or "usage" in captured.out.lower()

    def test_registry_command(self, capsys):
        main(["registry"])
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_unknown_command_shows_help(self, capsys):
        # argparse will error on truly unknown subcommands
        with pytest.raises(SystemExit):
            main(["nonexistent"])

    def test_download_requires_dataset(self):
        with pytest.raises(SystemExit):
            main(["download"])

    def test_model_requires_input(self):
        with pytest.raises(SystemExit):
            main(["model"])

    def test_report_requires_dirs(self):
        with pytest.raises(SystemExit):
            main(["report"])

    def test_standardize_requires_dataset(self):
        with pytest.raises(SystemExit):
            main(["standardize"])

    def test_features_requires_input(self):
        with pytest.raises(SystemExit):
            main(["features"])

    def test_repro_command_invokes_runner(self, monkeypatch):
        called = {}

        def _fake_run():
            called["ran"] = True
            return {"summary_report": Path("/tmp/repro.md")}

        monkeypatch.setattr("gender_gap.repro.run_repro_extension", _fake_run)
        main(["repro"])
        assert called["ran"] is True

    def test_variance_command_invokes_runner(self, monkeypatch, tmp_path: Path):
        called = {}

        def _fake_run(config_path=None):
            called["config_path"] = config_path
            return {"summary_report": tmp_path / "variance.md"}

        monkeypatch.setattr("gender_gap.variance.run_variance_addon", _fake_run)
        config_path = tmp_path / "variance.yaml"
        main(["variance", "--config", str(config_path)])
        assert called["config_path"] == config_path

    def test_download_sce_labor_market_writes_instructions(self, tmp_path: Path):
        main([
            "download",
            "--dataset", "sce_labor_market",
            "--output-dir", str(tmp_path),
        ])
        assert (tmp_path / "DOWNLOAD_INSTRUCTIONS.md").exists()

    def test_download_acs_api_replicate_weights_variant(self, monkeypatch, tmp_path: Path):
        calls = {}

        class DummyDownloader:
            def __init__(self, raw_dir):
                self.raw_dir = raw_dir

            def download(self, **kwargs):
                calls.update(kwargs)

        monkeypatch.setattr("gender_gap.downloaders.acs.ACSDownloader", DummyDownloader)
        main([
            "download",
            "--dataset", "acs",
            "--variant", "api",
            "--include-replicate-weights",
            "--years", "2023",
            "--output-dir", str(tmp_path),
        ])

        assert calls["mode"] == "api"
        assert calls["include_replicate_weights"] is True
        assert calls["years"] == [2023]

    def test_download_context_oews_variant_selects_oews_downloader(self, monkeypatch, tmp_path: Path):
        called = {}

        class DummyOEWSDownloader:
            def __init__(self, raw_dir):
                called["raw_dir"] = raw_dir

            def download(self, **kwargs):
                called.update(kwargs)

        monkeypatch.setattr("gender_gap.downloaders.context.OEWSDownloader", DummyOEWSDownloader)
        main([
            "download",
            "--dataset", "context",
            "--variant", "oews",
            "--years", "2023",
            "--output-dir", str(tmp_path),
        ])

        assert called["raw_dir"] == tmp_path
        assert called["years"] == [2023]

    def test_standardize_sipp_parquet(self, tmp_path: Path):
        raw = pd.DataFrame(
            {
                "SSUID": ["H1"],
                "PNUM": [1],
                "YEAR": [2023],
                "MONTHCODE": [1],
                "ESEX": [2],
                "RMESR": [1],
                "TJBHRS1": [40],
                "TJB1_MSUM": [5200],
            }
        )
        input_path = tmp_path / "sipp2023.parquet"
        output_path = tmp_path / "sipp_standardized.parquet"
        raw.to_parquet(input_path, index=False)

        main([
            "standardize",
            "--dataset", "sipp",
            "--input", str(input_path),
            "--output", str(output_path),
        ])

        assert output_path.exists()

        result = pd.read_parquet(output_path)
        assert result["calendar_year"].iloc[0] == 2023

    def test_standardize_sipp_public_use_filename_year(self, tmp_path: Path):
        raw = pd.DataFrame(
            {
                "SSUID": ["H1"],
                "PNUM": [1],
                "MONTHCODE": [1],
                "ESEX": [2],
                "RMESR": [1],
                "TJBHRS1": [40],
                "TJB1_MSUM": [5200],
            }
        )
        input_path = tmp_path / "pu2024.parquet"
        output_path = tmp_path / "sipp_standardized.parquet"
        raw.to_parquet(input_path, index=False)

        main([
            "standardize",
            "--dataset", "sipp",
            "--input", str(input_path),
            "--output", str(output_path),
        ])

        assert output_path.exists()

        result = pd.read_parquet(output_path)
        assert result["calendar_year"].iloc[0] == 2023

    def test_standardize_psid_uses_raw_dir(self, monkeypatch, tmp_path: Path):
        called = {}

        def _fake_standardize_psid_2023_for_gap(raw_dir=None):
            called["raw_dir"] = raw_dir
            return pd.DataFrame({"calendar_year": [2023], "female": [1]})

        monkeypatch.setattr(
            "gender_gap.standardize.psid_standardize.standardize_psid_2023_for_gap",
            _fake_standardize_psid_2023_for_gap,
        )

        output_path = tmp_path / "psid_standardized.parquet"
        main([
            "standardize",
            "--dataset", "psid",
            "--input", str(tmp_path),
            "--output", str(output_path),
        ])

        assert called["raw_dir"] == tmp_path
        assert output_path.exists()
