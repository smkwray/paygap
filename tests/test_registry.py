"""Tests for the dataset registry."""

from gender_gap.registry import core_datasets, get_dataset, load_registry


def test_load_registry_returns_entries():
    entries = load_registry()
    assert len(entries) > 0
    assert entries[0].dataset_id == "ACS_PUMS"


def test_core_datasets_filter():
    core = core_datasets()
    assert all(e.is_core for e in core)
    ids = {e.dataset_id for e in core}
    assert "ACS_PUMS" in ids
    assert "CPS_ORG_BASIC_MONTHLY" in ids
    assert "LAUS" in ids


def test_get_dataset_found():
    entry = get_dataset("ACS_PUMS")
    assert entry is not None
    assert entry.priority == "core"
    assert "commut" in entry.recommended_use.lower()


def test_get_dataset_not_found():
    assert get_dataset("NONEXISTENT_DATASET") is None


def test_get_sce_labor_market_dataset():
    entry = get_dataset("SCE_LABOR_MARKET")
    assert entry is not None
    assert entry.priority == "optional_support"
    assert "reservation wage" in entry.must_have_fields.lower()
