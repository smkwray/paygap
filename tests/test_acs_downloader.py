"""Tests for ACS downloader helper logic."""

from gender_gap.downloaders.acs import (
    ACS_REPLICATE_WEIGHT_COLUMNS,
    acs_api_variables,
    chunk_api_variables,
)


def test_acs_api_variables_add_replicate_weights():
    variables = acs_api_variables(2022, include_replicate_weights=True)

    assert "PWGTP" in variables
    assert "ESR" in variables
    assert "NOC" in variables
    assert "PAOC" in variables
    assert ACS_REPLICATE_WEIGHT_COLUMNS[0] in variables
    assert ACS_REPLICATE_WEIGHT_COLUMNS[-1] in variables
    assert "WKWN" in variables
    assert "WKW" not in variables


def test_acs_api_variables_pre_2019_use_legacy_names():
    variables = acs_api_variables(2018, include_replicate_weights=False)

    assert "WKW" in variables
    assert "JWTR" in variables
    assert "ESR" in variables
    assert "NOC" in variables
    assert "PAOC" in variables
    assert "WKWN" not in variables
    assert "JWTRNS" not in variables


def test_chunk_api_variables_repeats_merge_keys():
    variables = acs_api_variables(2022, include_replicate_weights=True)
    chunks = chunk_api_variables(variables, chunk_size=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk[:2] == ["SERIALNO", "SPORDER"]
        assert chunk.count("SERIALNO") == 1
        assert chunk.count("SPORDER") == 1
