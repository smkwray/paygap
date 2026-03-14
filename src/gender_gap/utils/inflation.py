"""Inflation adjustment utilities using CPI-U."""

from __future__ import annotations

from gender_gap.settings import BASE_CURRENCY_YEAR


def deflate_to_base_year(
    nominal_value: float,
    source_year: int,
    cpi_index: dict[int, float],
    base_year: int = BASE_CURRENCY_YEAR,
) -> float:
    """Deflate a nominal dollar amount to base-year dollars using CPI-U.

    Parameters
    ----------
    nominal_value : float
        The nominal dollar amount.
    source_year : int
        Year of the nominal value.
    cpi_index : dict[int, float]
        Mapping of year -> CPI-U annual average index.
    base_year : int
        Target base year for real dollars.

    Returns
    -------
    float
        Real dollar amount in base_year dollars.
    """
    if source_year not in cpi_index or base_year not in cpi_index:
        raise ValueError(
            f"CPI index missing year(s): source={source_year}, base={base_year}"
        )
    return nominal_value * (cpi_index[base_year] / cpi_index[source_year])
