"""Tests for year-aware harmonized occupation mapping."""

import pandas as pd

from gender_gap.crosswalks.occupation_crosswalks import (
    _build_harmonized_lookup,
    attach_harmonized_occupation_metadata,
)


def _sample_2018_list() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "occupation_code": "0500",
                "occupation_title_2018": (
                    "Agents and business managers of artists, performers, and athletes"
                ),
                "soc_code_2018": "13-1011",
            },
            {
                "occupation_code": "0051",
                "occupation_title_2018": "Marketing managers",
                "soc_code_2018": "11-2021",
            },
            {
                "occupation_code": "0052",
                "occupation_title_2018": "Sales managers",
                "soc_code_2018": "11-2022",
            },
            {
                "occupation_code": "0101",
                "occupation_title_2018": "Administrative services managers",
                "soc_code_2018": "11-3012",
            },
            {
                "occupation_code": "0202",
                "occupation_title_2018": "Compensation and benefits managers",
                "soc_code_2018": "11-3111",
            },
        ]
    )


def _sample_relations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "occupation_code_2010": "0100",
                "occupation_title_2010": "Administrative Services Managers",
                "soc_code_2010": "11-3011",
                "occupation_code_2018": "0101",
                "occupation_title_2018": "Administrative services managers",
                "soc_code_2018": "11-3012",
            },
            {
                "occupation_code_2010": "0200",
                "occupation_title_2010": "Benefits managers legacy A",
                "soc_code_2010": "11-3111",
                "occupation_code_2018": "0202",
                "occupation_title_2018": "Compensation and benefits managers",
                "soc_code_2018": "11-3111",
            },
            {
                "occupation_code_2010": "0201",
                "occupation_title_2010": "Benefits managers legacy B",
                "soc_code_2010": "11-3111",
                "occupation_code_2018": "0202",
                "occupation_title_2018": "Compensation and benefits managers",
                "soc_code_2018": "11-3111",
            },
            {
                "occupation_code_2010": "0300",
                "occupation_title_2010": "Marketing and Sales Managers",
                "soc_code_2010": "11-2020",
                "occupation_code_2018": "0051",
                "occupation_title_2018": "Marketing managers",
                "soc_code_2018": "11-2021",
            },
            {
                "occupation_code_2010": "0300",
                "occupation_title_2010": "Marketing and Sales Managers",
                "soc_code_2010": "11-2020",
                "occupation_code_2018": "0052",
                "occupation_title_2018": "Sales managers",
                "soc_code_2018": "11-2022",
            },
        ]
    )


def _sample_known_2010() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "occupation_code_2010": "0100",
                "occupation_title_2010": "Administrative Services Managers",
                "soc_code_2010": "11-3011",
            },
            {
                "occupation_code_2010": "0200",
                "occupation_title_2010": "Benefits managers legacy A",
                "soc_code_2010": "11-3111",
            },
            {
                "occupation_code_2010": "0201",
                "occupation_title_2010": "Benefits managers legacy B",
                "soc_code_2010": "11-3111",
            },
            {
                "occupation_code_2010": "0300",
                "occupation_title_2010": "Marketing and Sales Managers",
                "soc_code_2010": "11-2020",
            },
            {
                "occupation_code_2010": "0400",
                "occupation_title_2010": "Unmapped legacy role",
                "soc_code_2010": "11-9000",
            },
            {
                "occupation_code_2010": "0500",
                "occupation_title_2010": "Marketing and Sales Managers",
                "soc_code_2010": "11-2020",
            },
        ]
    )


def test_build_harmonized_lookup_is_year_aware():
    lookup = _build_harmonized_lookup(
        list_2018=_sample_2018_list(),
        relations=_sample_relations(),
        known_2010=_sample_known_2010(),
    )

    pre_0500 = lookup.loc[
        (lookup["occupation_code_raw"] == "0500")
        & (lookup["occupation_code_vintage"] == "2010")
    ].iloc[0]
    post_0500 = lookup.loc[
        (lookup["occupation_code_raw"] == "0500")
        & (lookup["occupation_code_vintage"] == "2018")
    ].iloc[0]

    assert pre_0500["occupation_harmonization_type"] == "legacy_2010_only"
    assert pre_0500["occupation_harmonized_code"] == "legacy_2010_0500"
    assert post_0500["occupation_harmonization_type"] == "native_2018"
    assert post_0500["occupation_harmonized_code"] == "0500"


def test_build_harmonized_lookup_semantic_types():
    lookup = _build_harmonized_lookup(
        list_2018=_sample_2018_list(),
        relations=_sample_relations(),
        known_2010=_sample_known_2010(),
    )
    keyed = lookup.set_index(["occupation_code_vintage", "occupation_code_raw"])

    assert keyed.loc[("2010", "0100"), "occupation_harmonization_type"] == "crosswalk_1_to_1"
    assert keyed.loc[("2010", "0200"), "occupation_harmonization_type"] == "crosswalk_many_to_1"
    assert keyed.loc[("2010", "0201"), "occupation_harmonization_type"] == "crosswalk_many_to_1"
    assert (
        keyed.loc[("2010", "0300"), "occupation_harmonization_type"]
        == "crosswalk_1_to_many_split_bucket"
    )
    assert keyed.loc[("2010", "0400"), "occupation_harmonization_type"] == "legacy_2010_only"


def test_attach_harmonized_occupation_metadata_uses_survey_year():
    lookup = _build_harmonized_lookup(
        list_2018=_sample_2018_list(),
        relations=_sample_relations(),
        known_2010=_sample_known_2010(),
    )
    data = pd.DataFrame(
        [
            {"occupation_code": 500, "survey_year": 2017},
            {"occupation_code": 500, "survey_year": 2018},
        ]
    )

    from gender_gap.crosswalks import occupation_crosswalks as occ

    original = occ.load_census_harmonized_occupation_lookup
    occ.load_census_harmonized_occupation_lookup = lambda path=None: lookup
    try:
        result = attach_harmonized_occupation_metadata(data, path=None)
    finally:
        occ.load_census_harmonized_occupation_lookup = original

    assert result.loc[0, "occupation_harmonized_code"] == "legacy_2010_0500"
    assert result.loc[0, "occupation_harmonization_type"] == "legacy_2010_only"
    assert result.loc[1, "occupation_harmonized_code"] == "0500"
    assert result.loc[1, "occupation_harmonization_type"] == "native_2018"
