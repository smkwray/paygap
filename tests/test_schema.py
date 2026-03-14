"""Tests for canonical schema definitions."""

from gender_gap.standardize.schema import (
    CONTEXT_AREA_TIME_COLUMNS,
    PERSON_DAY_TIMEUSE_COLUMNS,
    PERSON_MONTH_CORE_COLUMNS,
    PERSON_YEAR_CORE_COLUMNS,
)


def test_person_year_core_has_required_columns():
    required = {"person_id", "female", "hourly_wage_real", "person_weight", "data_source"}
    assert required.issubset(set(PERSON_YEAR_CORE_COLUMNS))


def test_person_month_core_has_overtime():
    assert "overtime_indicator" in PERSON_MONTH_CORE_COLUMNS


def test_person_day_timeuse_has_childcare():
    assert "minutes_childcare" in PERSON_DAY_TIMEUSE_COLUMNS


def test_context_has_unemployment():
    assert "local_unemployment_rate" in CONTEXT_AREA_TIME_COLUMNS


def test_no_duplicate_columns():
    for name, cols in [
        ("person_year_core", PERSON_YEAR_CORE_COLUMNS),
        ("person_month_core", PERSON_MONTH_CORE_COLUMNS),
        ("person_day_timeuse", PERSON_DAY_TIMEUSE_COLUMNS),
        ("context_area_time", CONTEXT_AREA_TIME_COLUMNS),
    ]:
        assert len(cols) == len(set(cols)), f"Duplicate columns in {name}"
