"""Household-level enrichment features.

This module adds partner and household-composition variables by linking
person records within the same ACS household (SERIALNO).  It is designed
as an *additive* enrichment step: the main pipeline continues to work
without it, and models that want these columns can opt in.

Expects a **standardized** ACS frame (after ``acs_standardize``), which
uses lowercase column names like ``acs_serialno``, ``acs_sporder``,
``wage_salary_income_real``, ``annual_earnings_real``, ``age``, etc.

New columns produced
--------------------
partner_wage_real     Partner's real wage/salary income, or NaN if no
                      partner/spouse identified in the household.
partner_earnings_real Partner's real total person earnings, or NaN.
relative_earnings     Respondent's wage / (respondent wage + partner wage).
                      NaN when partner is absent or both wages are zero.
partner_employed      1 if partner has positive wages, 0 if partner present
                      but wages <= 0, NaN if no partner identified.
multigenerational     1 if MULTG == 2 (Census coding: 1=No, 2=Yes),
                      0 otherwise, NaN if MULTG unavailable.
other_adults_present  Count of other adults (age >= 18) in the household,
                      excluding both the respondent and partner.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The standardized ACS frame stores both RELSHIPP (2019+) and RELP (pre-2019)
# values under a single ``relshipp`` column (see acs_standardize.py line 66).
# We must recognize BOTH code sets since the column name alone does not
# distinguish the vintage.
#
# RELSHIPP codes (2019+):
#   20 = reference person (NOT a partner)
#   21 = opposite-sex husband/wife/spouse
#   22 = opposite-sex unmarried partner
#   23 = same-sex husband/wife/spouse
#   24 = same-sex unmarried partner
#
# RELP codes (pre-2019):
#   0 = reference person
#   1 = husband/wife
#   13 = unmarried partner
#
# The two code sets do not overlap for partner codes, so the union is safe.
_PARTNER_CODES = {21, 22, 23, 24, 1, 13}


def enrich_household_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add partner/household-composition columns to a standardized ACS frame.

    Parameters
    ----------
    df : DataFrame
        A standardized ACS person-year frame.  Must contain at least
        ``acs_serialno`` and ``acs_sporder``.  Should also contain
        ``wage_salary_income_real``, ``annual_earnings_real``, ``age``,
        and a relationship column (``relshipp`` or ``relp``).

    Returns
    -------
    DataFrame
        A copy of *df* with the new columns appended.  Rows without a
        linkable partner get NaN in the partner columns.
    """
    out = df.copy()

    # --- Identify partner records ---
    partner_df = _find_partners(out)

    if partner_df is not None and not partner_df.empty:
        # Pull partner earnings (already real-dollar in standardized frame)
        earn_map = {}
        if "wage_salary_income_real" in out.columns:
            earn_map["wage_salary_income_real"] = "partner_wage_real"
        if "annual_earnings_real" in out.columns:
            earn_map["annual_earnings_real"] = "partner_earnings_real"

        pull_cols = ["acs_serialno", "acs_sporder", "age"] + list(earn_map.keys())
        pull_cols = [c for c in pull_cols if c in out.columns]

        partner_earnings = out[pull_cols].rename(
            columns={"acs_sporder": "partner_sporder", "age": "partner_age", **earn_map}
        )

        merged = partner_df.merge(
            partner_earnings,
            left_on=["acs_serialno", "partner_sporder"],
            right_on=["acs_serialno", "partner_sporder"],
            how="left",
        )

        # Join back to the main frame
        join_cols = ["acs_serialno", "acs_sporder"] + [
            c for c in merged.columns if c.startswith("partner_")
        ]
        out = out.merge(merged[join_cols], on=["acs_serialno", "acs_sporder"], how="left")

        # Derived: relative earnings
        resp_wage = pd.to_numeric(
            out["wage_salary_income_real"] if "wage_salary_income_real" in out.columns
            else pd.Series(0, index=out.index),
            errors="coerce",
        ).fillna(0)
        part_wage = pd.to_numeric(
            out["partner_wage_real"] if "partner_wage_real" in out.columns
            else pd.Series(0, index=out.index),
            errors="coerce",
        ).fillna(0)
        total = resp_wage + part_wage
        out["relative_earnings"] = np.where(total > 0, resp_wage / total, np.nan)

        # Derived: partner employed
        if "partner_wage_real" in out.columns:
            out["partner_employed"] = np.where(
                out["partner_wage_real"].notna(),
                (pd.to_numeric(out["partner_wage_real"], errors="coerce") > 0).astype(float),
                np.nan,
            )
        else:
            out["partner_employed"] = np.nan
    else:
        for col in [
            "partner_wage_real", "partner_earnings_real",
            "relative_earnings", "partner_employed",
        ]:
            out[col] = np.nan

    # --- Multigenerational household ---
    # Census MULTG coding: 1 = No, 2 = Yes
    if "multg" in out.columns:
        out["multigenerational"] = (
            pd.to_numeric(out["multg"], errors="coerce") == 2
        ).astype(float)
    else:
        out["multigenerational"] = np.nan

    # --- Other adults present (excluding respondent and partner) ---
    out["other_adults_present"] = _count_other_adults(out, partner_df)

    n_linked = out.get("partner_wage_real", pd.Series(dtype=float)).notna().sum()
    logger.info(
        "household enrichment: %d/%d rows linked to a partner (%.1f%%)",
        n_linked, len(out), 100 * n_linked / max(len(out), 1),
    )

    return out


def _find_partners(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return a (acs_serialno, acs_sporder, partner_sporder) mapping.

    Strategy: within each household, the householder (sporder==1) is linked
    to the person record with a spouse/partner RELSHIPP/RELP code.
    """
    if "acs_serialno" not in df.columns or "acs_sporder" not in df.columns:
        logger.warning("household enrichment: missing acs_serialno/acs_sporder, skipping")
        return None

    # The standardized ACS frame stores both RELSHIPP and RELP values
    # under "relshipp".  Use a unified code set that covers both vintages.
    if "relshipp" not in df.columns:
        logger.warning("household enrichment: no relshipp column found, skipping")
        return None

    rel = pd.to_numeric(df["relshipp"], errors="coerce")

    # Householder is SPORDER==1 by ACS convention.
    # Their partner is the person with a spouse/partner relationship code.
    is_householder = df["acs_sporder"] == 1
    is_partner_of_hh = rel.isin(_PARTNER_CODES)

    # Build householder -> partner mapping
    hh_records = df.loc[is_householder, ["acs_serialno", "acs_sporder"]].copy()
    partner_records = df.loc[is_partner_of_hh, ["acs_serialno", "acs_sporder"]].copy()

    if partner_records.empty:
        return None

    # Each household should have at most one partner record
    partner_records = partner_records.drop_duplicates(subset=["acs_serialno"], keep="first")
    partner_records = partner_records.rename(columns={"acs_sporder": "partner_sporder"})

    # Forward link: householder -> partner
    forward = hh_records.merge(partner_records, on="acs_serialno", how="inner")

    # Reverse link: partner -> householder
    reverse = partner_records.rename(columns={"partner_sporder": "acs_sporder"}).copy()
    reverse["partner_sporder"] = 1  # householder is always sporder 1

    links = pd.concat([forward, reverse], ignore_index=True)
    links = links.drop_duplicates(subset=["acs_serialno", "acs_sporder"], keep="first")

    return links


def _count_other_adults(
    df: pd.DataFrame,
    partner_df: pd.DataFrame | None,
) -> pd.Series:
    """Count adults (age >= 18) in the household excluding respondent and partner.

    Subtracts 1 (for the respondent) from every row, plus 1 more for the
    partner only on rows where *this specific person* has a linked partner.
    A third adult in a partnered household is not penalized by someone
    else's partner link.
    """
    if "acs_serialno" not in df.columns or "age" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    age = pd.to_numeric(df["age"], errors="coerce")
    is_adult = age >= 18

    # Count all adults per household
    adult_counts = (
        df.loc[is_adult]
        .groupby("acs_serialno")
        .size()
        .rename("total_adults")
    )

    merged = df[["acs_serialno"]].merge(adult_counts, on="acs_serialno", how="left")
    total = merged["total_adults"].fillna(0)

    # Subtract 1 for the respondent (always an adult in the prime-age sample)
    subtract = pd.Series(1, index=df.index)

    # Subtract 1 more only for rows where THIS person has a linked partner
    if partner_df is not None and not partner_df.empty:
        partner_keys = set(
            zip(partner_df["acs_serialno"], partner_df["acs_sporder"])
        )
        has_own_partner = pd.Series(
            [
                (sn, sp) in partner_keys
                for sn, sp in zip(df["acs_serialno"], df["acs_sporder"])
            ],
            index=df.index,
        )
        subtract = subtract + has_own_partner.astype(int)

    result = (total - subtract).clip(lower=0)
    result.index = df.index
    return result
