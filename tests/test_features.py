"""Tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from gender_gap.features.commute import (
    commute_bin,
    commute_mode_group,
    flag_long_commute,
)
from gender_gap.features.earnings import (
    compute_hourly_wage,
    deflate_series,
    log_wage,
    winsorize_wages,
)
from gender_gap.features.family import (
    any_children,
    has_young_children,
    parenthood_category,
)
from gender_gap.features.sample_filters import (
    drop_outlier_wages,
    filter_all_employed,
    filter_prime_age_wage_salary,
)


class TestEarnings:
    def test_hourly_wage_annual(self):
        earnings = pd.Series([52000.0])
        hours = pd.Series([40.0])
        weeks = pd.Series([50.0])
        result = compute_hourly_wage(earnings, hours, weeks, method="annual")
        assert result.iloc[0] == pytest.approx(26.0)

    def test_hourly_wage_weekly(self):
        earnings = pd.Series([1000.0])
        hours = pd.Series([40.0])
        result = compute_hourly_wage(earnings, hours, method="weekly")
        assert result.iloc[0] == pytest.approx(25.0)

    def test_hourly_wage_zero_hours_nan(self):
        result = compute_hourly_wage(
            pd.Series([1000.0]), pd.Series([0.0]), method="weekly"
        )
        assert pd.isna(result.iloc[0])

    def test_winsorize_clips(self):
        wages = pd.Series([1.0, 10.0, 20.0, 30.0, 100.0])
        result = winsorize_wages(wages, lower_pct=10, upper_pct=90)
        assert result.min() >= wages.quantile(0.10) - 0.01
        assert result.max() <= wages.quantile(0.90) + 0.01

    def test_log_wage_positive(self):
        result = log_wage(pd.Series([10.0, 20.0]))
        assert result.iloc[0] == pytest.approx(np.log(10.0))

    def test_log_wage_nonpositive_nan(self):
        result = log_wage(pd.Series([0.0, -5.0, 10.0]))
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])

    def test_deflate_series(self):
        values = pd.Series([100.0, 100.0])
        years = pd.Series([2020, 2024])
        cpi = {2020: 258.8, 2024: 314.0}
        result = deflate_series(values, years, cpi, base_year=2024)
        assert result.iloc[0] > 100.0  # 2020 dollars should inflate
        assert result.iloc[1] == pytest.approx(100.0)  # same year


class TestCommute:
    def test_commute_bin_categories(self):
        minutes = pd.Series([0, 10, 25, 40, 55, 90])
        result = commute_bin(minutes)
        assert result.iloc[0] == "0"
        assert result.iloc[1] == "1-15"
        assert result.iloc[2] == "16-30"
        assert result.iloc[3] == "31-45"
        assert result.iloc[4] == "46-60"
        assert result.iloc[5] == "60+"

    def test_commute_mode_group_mapping(self):
        modes = pd.Series(["car_truck_van_alone", "bus", "walked", "work_from_home"])
        result = commute_mode_group(modes)
        assert result.iloc[0] == "drive_alone"
        assert result.iloc[1] == "transit"
        assert result.iloc[2] == "walk_bike"
        assert result.iloc[3] == "wfh"

    def test_long_commute_flag(self):
        minutes = pd.Series([10, 30, 50, 90])
        result = flag_long_commute(minutes, threshold=45)
        assert list(result) == [0, 0, 1, 1]


class TestFamily:
    def test_parenthood_category(self):
        nc = pd.Series([0, 2, 1])
        cu5 = pd.Series([0, 1, 0])
        result = parenthood_category(nc, cu5)
        assert result.iloc[0] == "no_children"
        assert result.iloc[1] == "young_children"
        assert result.iloc[2] == "has_children"

    def test_has_young_children(self):
        result = has_young_children(pd.Series([0, 1, 2]))
        assert list(result) == [0, 1, 1]

    def test_any_children(self):
        result = any_children(pd.Series([0, 1, 3]))
        assert list(result) == [0, 1, 1]


class TestSampleFilters:
    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame({
            "age": [22, 30, 40, 55, 35],
            "self_employed": [0, 0, 1, 0, 0],
            "usual_hours_week": [0, 40, 40, 35, 40],
            "hourly_wage_real": [15.0, 25.0, 30.0, 20.0, 28.0],
            "weeks_worked": [20, 52, 52, 52, 52],
            "work_from_home": [0, 0, 0, 0, 1],
            "commute_minutes_one_way": [20.0, 30.0, 15.0, 45.0, pd.NA],
            "employed": [1, 1, 1, 1, 1],
        })

    def test_prime_age_filter(self, sample_df):
        result = filter_prime_age_wage_salary(sample_df)
        # age 22 excluded (too young), 55 excluded (too old),
        # self_employed=1 excluded, hours=0 excluded
        assert len(result) == 2  # rows with age 30 and 35

    def test_all_employed_filter(self, sample_df):
        result = filter_all_employed(sample_df)
        # age 22 kept (18-64), 55 kept, but hours=0 excluded
        assert len(result) == 4

    def test_drop_outlier_wages(self, sample_df):
        sample_df.loc[0, "hourly_wage_real"] = 1.0  # below $2
        result = drop_outlier_wages(sample_df, min_wage=2.0)
        assert len(result) == 4
