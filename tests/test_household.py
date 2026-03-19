"""Tests for household-level enrichment features."""

import numpy as np
import pandas as pd
import pytest

from gender_gap.features.household import (
    _PARTNER_CODES,
    _count_other_adults,
    _find_partners,
    enrich_household_features,
)


def _make_household(*persons):
    """Build a standardized-schema DataFrame from (serialno, sporder, relshipp, age, wage, earnings) tuples."""
    rows = []
    for sn, sp, rel, age, wage, earn in persons:
        rows.append({
            "acs_serialno": str(sn),
            "acs_sporder": sp,
            "relshipp": rel,
            "age": age,
            "wage_salary_income_real": wage,
            "annual_earnings_real": earn,
            "multg": np.nan,
        })
    return pd.DataFrame(rows)


class TestPartnerCodes:
    """Verify the unified code set covers both ACS vintages."""

    def test_relshipp_codes_present(self):
        # 2019+ spouse/partner codes
        for code in [21, 22, 23, 24]:
            assert code in _PARTNER_CODES

    def test_relp_codes_present(self):
        # Pre-2019 spouse/partner codes
        for code in [1, 13]:
            assert code in _PARTNER_CODES

    def test_reference_person_excluded(self):
        # RELSHIPP 20 = reference person, RELP 0 = reference person
        assert 20 not in _PARTNER_CODES
        assert 0 not in _PARTNER_CODES


class TestFindPartners:
    def test_opposite_sex_spouse_relshipp(self):
        """Householder + opposite-sex spouse (2019+ RELSHIPP=21)."""
        df = _make_household(
            ("HH1", 1, 20, 35, 60000, 60000),  # householder
            ("HH1", 2, 21, 33, 45000, 45000),  # opp-sex spouse
        )
        links = _find_partners(df)
        assert links is not None
        assert len(links) == 2  # bidirectional
        # Householder -> partner
        hh_row = links[links["acs_sporder"] == 1].iloc[0]
        assert hh_row["partner_sporder"] == 2
        # Partner -> householder
        pt_row = links[links["acs_sporder"] == 2].iloc[0]
        assert pt_row["partner_sporder"] == 1

    def test_same_sex_partner_relshipp(self):
        """Same-sex unmarried partner (2019+ RELSHIPP=24)."""
        df = _make_household(
            ("HH2", 1, 20, 40, 50000, 50000),
            ("HH2", 2, 24, 38, 55000, 55000),
        )
        links = _find_partners(df)
        assert links is not None
        assert len(links) == 2

    def test_pre2019_relp_spouse(self):
        """Pre-2019 husband/wife (RELP=1 stored under relshipp column)."""
        df = _make_household(
            ("HH3", 1, 0, 42, 70000, 70000),   # reference person (RELP=0)
            ("HH3", 2, 1, 40, 50000, 50000),   # husband/wife (RELP=1)
        )
        links = _find_partners(df)
        assert links is not None
        assert len(links) == 2

    def test_pre2019_relp_unmarried_partner(self):
        """Pre-2019 unmarried partner (RELP=13 stored under relshipp column)."""
        df = _make_household(
            ("HH4", 1, 0, 30, 40000, 40000),
            ("HH4", 2, 13, 28, 35000, 35000),
        )
        links = _find_partners(df)
        assert links is not None
        assert len(links) == 2

    def test_no_partner(self):
        """Single-person household — no partner found."""
        df = _make_household(
            ("HH5", 1, 20, 45, 80000, 80000),
        )
        links = _find_partners(df)
        assert links is None

    def test_child_not_matched_as_partner(self):
        """Adult child (RELSHIPP=25) should not be matched."""
        df = _make_household(
            ("HH6", 1, 20, 50, 60000, 60000),
            ("HH6", 2, 25, 22, 20000, 20000),  # biological child
        )
        links = _find_partners(df)
        assert links is None

    def test_multiple_households(self):
        """Two separate households, each with a partner."""
        df = pd.concat([
            _make_household(
                ("A", 1, 20, 35, 60000, 60000),
                ("A", 2, 21, 33, 45000, 45000),
            ),
            _make_household(
                ("B", 1, 0, 40, 70000, 70000),
                ("B", 2, 1, 38, 50000, 50000),
            ),
        ], ignore_index=True)
        links = _find_partners(df)
        assert links is not None
        assert len(links) == 4  # 2 per household

    def test_missing_relshipp_column(self):
        """No relationship column at all — returns None."""
        df = pd.DataFrame({
            "acs_serialno": ["X"],
            "acs_sporder": [1],
            "age": [30],
        })
        links = _find_partners(df)
        assert links is None


class TestEnrichHouseholdFeatures:
    def test_partner_wages_linked(self):
        """Partner wages should appear on both householder and partner rows."""
        df = _make_household(
            ("HH1", 1, 20, 35, 60000, 65000),
            ("HH1", 2, 21, 33, 45000, 48000),
        )
        result = enrich_household_features(df)
        # Householder sees partner's wage
        hh = result[result["acs_sporder"] == 1].iloc[0]
        assert hh["partner_wage_real"] == pytest.approx(45000)
        assert hh["partner_earnings_real"] == pytest.approx(48000)
        # Partner sees householder's wage
        pt = result[result["acs_sporder"] == 2].iloc[0]
        assert pt["partner_wage_real"] == pytest.approx(60000)

    def test_relative_earnings(self):
        """Relative earnings = respondent / (respondent + partner)."""
        df = _make_household(
            ("HH1", 1, 20, 35, 60000, 60000),
            ("HH1", 2, 21, 33, 40000, 40000),
        )
        result = enrich_household_features(df)
        hh = result[result["acs_sporder"] == 1].iloc[0]
        assert hh["relative_earnings"] == pytest.approx(60000 / 100000)
        pt = result[result["acs_sporder"] == 2].iloc[0]
        assert pt["relative_earnings"] == pytest.approx(40000 / 100000)

    def test_relative_earnings_both_zero(self):
        """Both wages zero → NaN."""
        df = _make_household(
            ("HH1", 1, 20, 35, 0, 0),
            ("HH1", 2, 21, 33, 0, 0),
        )
        result = enrich_household_features(df)
        assert pd.isna(result.iloc[0]["relative_earnings"])

    def test_partner_employed(self):
        """partner_employed = 1 if positive wage, 0 if zero, NaN if no partner."""
        df = pd.concat([
            _make_household(
                ("A", 1, 20, 35, 60000, 60000),
                ("A", 2, 21, 33, 45000, 45000),  # employed partner
            ),
            _make_household(
                ("B", 1, 20, 40, 50000, 50000),
                ("B", 2, 21, 38, 0, 0),           # non-employed partner
            ),
            _make_household(
                ("C", 1, 20, 45, 80000, 80000),   # no partner
            ),
        ], ignore_index=True)
        result = enrich_household_features(df)
        a_hh = result[(result["acs_serialno"] == "A") & (result["acs_sporder"] == 1)].iloc[0]
        assert a_hh["partner_employed"] == 1.0
        b_hh = result[(result["acs_serialno"] == "B") & (result["acs_sporder"] == 1)].iloc[0]
        assert b_hh["partner_employed"] == 0.0
        c_hh = result[(result["acs_serialno"] == "C") & (result["acs_sporder"] == 1)].iloc[0]
        assert pd.isna(c_hh["partner_employed"])

    def test_no_partner_columns_nan(self):
        """Single-person household gets NaN for all partner columns."""
        df = _make_household(("S", 1, 20, 30, 50000, 50000))
        result = enrich_household_features(df)
        row = result.iloc[0]
        assert pd.isna(row["partner_wage_real"])
        assert pd.isna(row["partner_earnings_real"])
        assert pd.isna(row["relative_earnings"])
        assert pd.isna(row["partner_employed"])


class TestPreTwentyNineteenEnrichment:
    """End-to-end enrichment with pre-2019 RELP codes stored under relshipp."""

    def test_relp_spouse_wages_flow_bidirectionally(self):
        """RELP=1 (husband/wife) stored under relshipp column: both sides get partner wages."""
        df = _make_household(
            ("HH", 1, 0, 42, 70000, 75000),   # reference person (RELP=0)
            ("HH", 2, 1, 40, 50000, 55000),   # husband/wife (RELP=1)
        )
        result = enrich_household_features(df)
        hh = result[result["acs_sporder"] == 1].iloc[0]
        pt = result[result["acs_sporder"] == 2].iloc[0]
        # Householder sees partner's wage
        assert hh["partner_wage_real"] == pytest.approx(50000)
        assert hh["partner_earnings_real"] == pytest.approx(55000)
        # Partner sees householder's wage
        assert pt["partner_wage_real"] == pytest.approx(70000)
        assert pt["partner_earnings_real"] == pytest.approx(75000)

    def test_relp_unmarried_partner_wages_flow(self):
        """RELP=13 (unmarried partner) stored under relshipp column."""
        df = _make_household(
            ("HH", 1, 0, 30, 40000, 42000),
            ("HH", 2, 13, 28, 35000, 37000),
        )
        result = enrich_household_features(df)
        hh = result[result["acs_sporder"] == 1].iloc[0]
        assert hh["partner_wage_real"] == pytest.approx(35000)
        pt = result[result["acs_sporder"] == 2].iloc[0]
        assert pt["partner_wage_real"] == pytest.approx(40000)


class TestMissingColumns:
    def test_missing_wage_column(self):
        """If wage_salary_income_real is absent, partner_wage_real should be NaN."""
        df = _make_household(
            ("HH", 1, 20, 35, 60000, 60000),
            ("HH", 2, 21, 33, 45000, 45000),
        )
        df.drop(columns=["wage_salary_income_real"], inplace=True)
        result = enrich_household_features(df)
        # Should not crash; partner_wage_real won't exist
        assert "partner_wage_real" not in result.columns or pd.isna(result.iloc[0].get("partner_wage_real"))

    def test_missing_earnings_column(self):
        """If annual_earnings_real is absent, partner_earnings_real should be NaN."""
        df = _make_household(
            ("HH", 1, 20, 35, 60000, 60000),
            ("HH", 2, 21, 33, 45000, 45000),
        )
        df.drop(columns=["annual_earnings_real"], inplace=True)
        result = enrich_household_features(df)
        assert "partner_earnings_real" not in result.columns or pd.isna(result.iloc[0].get("partner_earnings_real"))

    def test_missing_serialno(self):
        """Missing acs_serialno entirely — should return frame with NaN partner cols."""
        df = pd.DataFrame({
            "acs_sporder": [1],
            "relshipp": [20],
            "age": [30],
            "wage_salary_income_real": [50000],
            "annual_earnings_real": [50000],
            "multg": [1],
        })
        result = enrich_household_features(df)
        assert pd.isna(result.iloc[0]["partner_wage_real"])

    def test_duplicate_partner_records(self):
        """Two people coded as partners in same household — only first is used."""
        df = _make_household(
            ("HH", 1, 20, 35, 60000, 60000),
            ("HH", 2, 21, 33, 45000, 45000),  # first spouse
            ("HH", 3, 22, 30, 30000, 30000),  # second "partner" — should be ignored
        )
        links = _find_partners(df)
        assert links is not None
        hh_links = links[links["acs_sporder"] == 1]
        assert len(hh_links) == 1
        assert hh_links.iloc[0]["partner_sporder"] == 2  # first one wins


class TestMultigenerational:
    def test_multg_yes(self):
        """MULTG == 2 → multigenerational = 1."""
        df = _make_household(("M", 1, 20, 35, 50000, 50000))
        df["multg"] = 2
        result = enrich_household_features(df)
        assert result.iloc[0]["multigenerational"] == 1.0

    def test_multg_no(self):
        """MULTG == 1 → multigenerational = 0."""
        df = _make_household(("M", 1, 20, 35, 50000, 50000))
        df["multg"] = 1
        result = enrich_household_features(df)
        assert result.iloc[0]["multigenerational"] == 0.0

    def test_multg_missing(self):
        """No multg column → multigenerational = NaN."""
        df = _make_household(("M", 1, 20, 35, 50000, 50000))
        df.drop(columns=["multg"], inplace=True)
        result = enrich_household_features(df)
        assert pd.isna(result.iloc[0]["multigenerational"])


class TestOtherAdultsPresent:
    def test_couple_only(self):
        """Two-adult couple: each sees 0 other adults."""
        df = _make_household(
            ("HH", 1, 20, 35, 60000, 60000),
            ("HH", 2, 21, 33, 45000, 45000),
        )
        result = enrich_household_features(df)
        assert result.iloc[0]["other_adults_present"] == 0
        assert result.iloc[1]["other_adults_present"] == 0

    def test_couple_plus_adult_child(self):
        """Couple + one adult child: each coupled person sees 1 other adult.
        The adult child also sees 1 (their parent's partner is not their partner)."""
        df = _make_household(
            ("HH", 1, 20, 50, 60000, 60000),  # householder
            ("HH", 2, 21, 48, 45000, 45000),  # spouse
            ("HH", 3, 25, 22, 20000, 20000),  # adult child (not a partner)
        )
        result = enrich_household_features(df)
        # Householder: 3 adults - 1 (self) - 1 (spouse) = 1
        assert result[result["acs_sporder"] == 1].iloc[0]["other_adults_present"] == 1
        # Spouse: 3 adults - 1 (self) - 1 (householder) = 1
        assert result[result["acs_sporder"] == 2].iloc[0]["other_adults_present"] == 1
        # Adult child: 3 adults - 1 (self) - 0 (no own partner) = 2
        assert result[result["acs_sporder"] == 3].iloc[0]["other_adults_present"] == 2

    def test_single_person(self):
        """One adult, no partner: 0 other adults."""
        df = _make_household(("S", 1, 20, 30, 50000, 50000))
        result = enrich_household_features(df)
        assert result.iloc[0]["other_adults_present"] == 0

    def test_minor_child_not_counted(self):
        """Children under 18 should not be counted as other adults."""
        df = _make_household(
            ("HH", 1, 20, 40, 60000, 60000),
            ("HH", 2, 21, 38, 45000, 45000),
            ("HH", 3, 25, 15, 0, 0),  # minor child
        )
        result = enrich_household_features(df)
        assert result[result["acs_sporder"] == 1].iloc[0]["other_adults_present"] == 0
