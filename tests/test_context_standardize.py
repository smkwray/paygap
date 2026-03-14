import pandas as pd

from gender_gap.standardize.context_standardize import (
    standardize_bea_rpp,
    standardize_oews,
    standardize_qcew,
)


def test_standardize_qcew_coerces_area_fips_and_numeric_values():
    df = pd.DataFrame({
        "area_fips": ["1001", "06001.0"],
        "year": ["2023", "2022"],
        "annual_avg_wkly_wage": ["1100", "1250.5"],
        "annual_avg_emplvl": ["100", "250"],
    })

    out = standardize_qcew(df)

    assert list(out["geography_key"]) == ["01001", "06001"]
    assert list(out["calendar_year"]) == [2023, 2022]
    assert list(out["local_industry_avg_weekly_wage"]) == [1100.0, 1250.5]


def test_standardize_oews_uses_all_occupations_and_lowercase_columns():
    df = pd.DataFrame({
        "area": ["01", "01"],
        "year": [2023, 2023],
        "occ_code": ["00-0000", "11-1011"],
        "a_mean": ["60000", "120000"],
    })

    out = standardize_oews(df)

    assert len(out) == 1
    assert out.iloc[0]["geography_key"] == "01"
    assert out.iloc[0]["local_occupation_avg_wage"] == 60000.0


def test_standardize_bea_rpp_coerces_geo_and_rpp():
    df = pd.DataFrame({
        "GeoFips": ["01000"],
        "Year": ["2023"],
        "RPP": ["97.2"],
    })

    out = standardize_bea_rpp(df)

    assert out.iloc[0]["geography_key"] == "01000"
    assert out.iloc[0]["calendar_year"] == 2023
    assert out.iloc[0]["local_price_parity"] == 97.2
