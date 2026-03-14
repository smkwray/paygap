"""Family and parenthood feature engineering."""

from __future__ import annotations

import pandas as pd


def parenthood_category(
    num_children: pd.Series,
    children_under_5: pd.Series,
) -> pd.Series:
    """Classify parenthood status.

    Categories: no_children, young_children, school_age_only, teenagers_only
    """
    result = pd.Series("no_children", index=num_children.index)
    result[num_children > 0] = "has_children"
    result[children_under_5 > 0] = "young_children"
    return result


def has_young_children(children_under_5: pd.Series) -> pd.Series:
    """Binary indicator for presence of children under 5."""
    return (children_under_5 > 0).astype(int)


def any_children(num_children: pd.Series) -> pd.Series:
    """Binary indicator for presence of any children."""
    return (num_children > 0).astype(int)
