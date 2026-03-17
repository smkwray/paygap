"""Tests for crosswalk modules: occupation, industry, geography."""

from pathlib import Path

import pandas as pd

from gender_gap.crosswalks.geography_crosswalks import (
    append_puma_cbsa_crosswalk,
    build_geo_merge_key,
    load_puma_cbsa_crosswalk,
    metro_indicator,
    state_fips_to_abbr,
    state_fips_to_region,
    state_fips_to_region_label,
)
from gender_gap.crosswalks.industry_crosswalks import (
    census_ind_to_naics2,
    naics2_to_broad,
    naics2_to_label,
)
from gender_gap.crosswalks.occupation_crosswalks import (
    census_occ_to_soc_major,
    onet_soc_to_census_soc,
    soc_major_to_broad,
    soc_major_to_label,
)

# --- Occupation crosswalks ---

class TestOccupationCrosswalks:
    def test_management_range(self):
        codes = pd.Series([10, 200, 440])
        result = census_occ_to_soc_major(codes)
        assert (result == "11").all()

    def test_computer_math_range(self):
        codes = pd.Series([1005, 1100, 1240])
        result = census_occ_to_soc_major(codes)
        assert (result == "15").all()

    def test_unknown_code(self):
        codes = pd.Series([9999, -1])
        result = census_occ_to_soc_major(codes)
        assert (result == "unknown").all()

    def test_soc_major_to_label(self):
        soc2 = pd.Series(["11", "15", "29"])
        result = soc_major_to_label(soc2)
        assert result.iloc[0] == "Management"
        assert result.iloc[1] == "Computer and Mathematical"
        assert result.iloc[2] == "Healthcare Practitioners and Technical"

    def test_soc_major_to_label_unknown(self):
        soc2 = pd.Series(["99", "unknown"])
        result = soc_major_to_label(soc2)
        assert (result == "Unknown").all()

    def test_soc_major_to_broad(self):
        soc2 = pd.Series(["11", "29", "35", "41", "51"])
        result = soc_major_to_broad(soc2)
        assert result.iloc[0] == "management_professional"
        assert result.iloc[1] == "healthcare"
        assert result.iloc[2] == "service"
        assert result.iloc[3] == "sales_office"
        assert result.iloc[4] == "production_transport"

    def test_onet_soc_to_census_soc(self):
        onet = pd.Series(["11-1011.00", "15-1252.00", "29-1141.00"])
        result = onet_soc_to_census_soc(onet)
        assert list(result) == ["11", "15", "29"]

    def test_multiple_occupation_ranges(self):
        codes = pd.Series([500, 3000, 4700, 6200, 9000])
        result = census_occ_to_soc_major(codes)
        expected = ["13", "29", "41", "47", "53"]
        assert list(result) == expected


# --- Industry crosswalks ---

class TestIndustryCrosswalks:
    def test_manufacturing_range(self):
        codes = pd.Series([1070, 2000, 3990])
        result = census_ind_to_naics2(codes)
        assert (result == "31").all()

    def test_retail_range(self):
        codes = pd.Series([4670, 5000, 5790])
        result = census_ind_to_naics2(codes)
        assert (result == "44").all()

    def test_unknown_industry(self):
        codes = pd.Series([0, 99999])
        result = census_ind_to_naics2(codes)
        assert (result == "unknown").all()

    def test_naics2_to_label(self):
        naics2 = pd.Series(["31", "52", "62"])
        result = naics2_to_label(naics2)
        assert result.iloc[0] == "Manufacturing"
        assert result.iloc[1] == "Finance and Insurance"
        assert result.iloc[2] == "Health Care and Social Assistance"

    def test_naics2_to_broad(self):
        naics2 = pd.Series(["31", "52", "62", "72"])
        result = naics2_to_broad(naics2)
        assert result.iloc[0] == "manufacturing"
        assert result.iloc[1] == "information_finance"
        assert result.iloc[2] == "education_health"
        assert result.iloc[3] == "leisure_hospitality"

    def test_agriculture_range(self):
        codes = pd.Series([170, 230, 290])
        result = census_ind_to_naics2(codes)
        assert (result == "11").all()


# --- Geography crosswalks ---

class TestGeographyCrosswalks:
    def test_state_fips_to_abbr(self):
        fips = pd.Series([6, 36, 48])
        result = state_fips_to_abbr(fips)
        assert list(result) == ["CA", "NY", "TX"]

    def test_state_fips_to_abbr_unknown(self):
        fips = pd.Series([99, None])
        result = state_fips_to_abbr(fips)
        assert (result == "UNK").all()

    def test_state_fips_to_region(self):
        fips = pd.Series([6, 17, 36, 48])
        result = state_fips_to_region(fips)
        assert list(result) == ["WE", "MW", "NE", "SO"]

    def test_state_fips_to_region_label(self):
        fips = pd.Series([6, 17])
        result = state_fips_to_region_label(fips)
        assert list(result) == ["West", "Midwest"]

    def test_build_geo_merge_key_state_only(self):
        df = pd.DataFrame({"state_fips": [6, 36]})
        result = build_geo_merge_key(df)
        assert result.iloc[0] == "state:6"
        assert result.iloc[1] == "state:36"

    def test_build_geo_merge_key_fallback(self):
        df = pd.DataFrame({
            "state_fips": [6, 36],
            "county_fips": ["06037", None],
        })
        result = build_geo_merge_key(df)
        assert result.iloc[0] == "county:06037"
        assert result.iloc[1] == "state:36"

    def test_build_geo_merge_key_puma_with_state(self):
        df = pd.DataFrame({
            "state_fips": [6],
            "residence_puma": [3700],
        })
        result = build_geo_merge_key(df)
        assert result.iloc[0] == "puma:6_3700"

    def test_build_geo_merge_key_national_fallback(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        result = build_geo_merge_key(df)
        assert (result == "national:US").all()

    def test_metro_indicator_placeholder(self):
        fips = pd.Series([6, 36])
        result = metro_indicator(fips)
        assert (result == "unknown").all()

    def test_load_and_append_puma_cbsa_crosswalk(self, tmp_path: Path):
        crosswalk = pd.DataFrame(
            {
                "state_fips": ["06", "36"],
                "puma": ["03700", "03800"],
                "cbsa_code": ["31080", "35620"],
                "cbsa_title": [
                    "Los Angeles-Long Beach-Anaheim, CA",
                    "New York-Newark-Jersey City, NY-NJ-PA",
                ],
                "metro_status": ["metropolitan", "metropolitan"],
                "dominant_overlap_area": [1.0, 2.0],
            }
        )
        path = tmp_path / "puma_to_cbsa_crosswalk.csv"
        crosswalk.to_csv(path, index=False)

        loaded = load_puma_cbsa_crosswalk(path=path, force_reload=True)
        assert loaded is not None
        assert loaded["puma"].tolist() == ["03700", "03800"]

        df = pd.DataFrame(
            {
                "state_fips": [6, 36, 48],
                "residence_puma": [3700, 3800, 1900],
            }
        )
        result = append_puma_cbsa_crosswalk(df, path=path)
        assert result["cbsa_code"].iloc[0] == "31080"
        assert result["cbsa_code"].iloc[1] == "35620"
        assert pd.isna(result["cbsa_code"].iloc[2])
        assert result["metro_status"].tolist() == ["metropolitan", "metropolitan", "unknown"]

    def test_metro_indicator_uses_crosswalk_when_available(self, tmp_path: Path):
        pd.DataFrame(
            {
                "state_fips": ["06"],
                "puma": ["03700"],
                "cbsa_code": ["31080"],
                "cbsa_title": ["Los Angeles-Long Beach-Anaheim, CA"],
                "metro_status": ["metropolitan"],
                "dominant_overlap_area": [1.0],
            }
        ).to_csv(tmp_path / "puma_to_cbsa_crosswalk.csv", index=False)

        result = metro_indicator(
            pd.Series([6, 48]),
            pd.Series([3700, 1900]),
            path=tmp_path / "puma_to_cbsa_crosswalk.csv",
        )
        assert result.tolist() == ["metropolitan", "unknown"]

    def test_dc_and_pr(self):
        fips = pd.Series([11, 72])
        result = state_fips_to_abbr(fips)
        assert list(result) == ["DC", "PR"]
        regions = state_fips_to_region(fips)
        assert list(regions) == ["SO", "PR"]
