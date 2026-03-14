"""Commute feature engineering.

Builds commute bins, mode categories, and work-from-home indicators.
"""

from __future__ import annotations

import pandas as pd


def commute_bin(minutes: pd.Series) -> pd.Series:
    """Bin one-way commute minutes into standard categories.

    Bins: 0 (WFH/NA), 1-15, 16-30, 31-45, 46-60, 60+
    """
    bins = [-1, 0, 15, 30, 45, 60, float("inf")]
    labels = ["0", "1-15", "16-30", "31-45", "46-60", "60+"]
    return pd.cut(minutes.fillna(-0.5), bins=bins, labels=labels, right=True)


def commute_mode_group(mode: pd.Series) -> pd.Series:
    """Group detailed commute modes into broad categories.

    Returns: drive_alone, carpool, transit, walk_bike, wfh, other
    """
    mapping = {
        "car_truck_van_alone": "drive_alone",
        "car_truck_van_carpool": "carpool",
        "bus": "transit",
        "streetcar_trolley": "transit",
        "subway_elevated": "transit",
        "railroad": "transit",
        "ferryboat": "transit",
        "taxicab": "other",
        "motorcycle": "drive_alone",
        "bicycle": "walk_bike",
        "walked": "walk_bike",
        "work_from_home": "wfh",
    }
    return mode.map(mapping).fillna("other")


def flag_long_commute(minutes: pd.Series, threshold: int = 45) -> pd.Series:
    """Flag commuters with one-way commute above threshold minutes."""
    return (minutes > threshold).astype(int)
