"""Validation helpers for data quality checks."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_required_columns(df, required: list[str], context: str = "") -> list[str]:
    """Check that all required columns are present. Return list of missing columns."""
    if hasattr(df, "columns"):
        present = set(df.columns)
    else:
        present = set()
    missing = [c for c in required if c not in present]
    if missing:
        logger.warning("Missing columns in %s: %s", context, missing)
    return missing


def check_no_all_null(df, columns: list[str], context: str = "") -> list[str]:
    """Return columns that are entirely null."""
    all_null = []
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].isna().all():
            all_null.append(col)
    if all_null:
        logger.warning("All-null columns in %s: %s", context, all_null)
    return all_null
