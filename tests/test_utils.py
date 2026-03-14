"""Tests for utility functions."""

import numpy as np
import pytest

from gender_gap.utils.inflation import deflate_to_base_year
from gender_gap.utils.weights import (
    confidence_interval,
    replicate_weight_columns,
    sdr_standard_error,
    weighted_mean,
    weighted_quantile,
)


class TestInflation:
    def test_same_year_returns_same_value(self):
        cpi = {2022: 292.655, 2024: 314.0}
        assert deflate_to_base_year(100.0, 2024, cpi, base_year=2024) == 100.0

    def test_deflation_direction(self):
        cpi = {2022: 292.655, 2024: 314.0}
        real = deflate_to_base_year(100.0, 2022, cpi, base_year=2024)
        assert real > 100.0  # 2024 dollars > 2022 dollars

    def test_missing_year_raises(self):
        cpi = {2022: 292.655}
        with pytest.raises(ValueError, match="CPI index missing"):
            deflate_to_base_year(100.0, 2022, cpi, base_year=2024)


class TestWeightedMean:
    def test_equal_weights(self):
        assert weighted_mean([1, 2, 3], [1, 1, 1]) == pytest.approx(2.0)

    def test_unequal_weights(self):
        result = weighted_mean([10, 20], [3, 1])
        assert result == pytest.approx(12.5)

    def test_nan_handling(self):
        result = weighted_mean([1, np.nan, 3], [1, 1, 1])
        assert result == pytest.approx(2.0)

    def test_all_nan_returns_nan(self):
        assert np.isnan(weighted_mean([np.nan], [1]))


class TestWeightedQuantile:
    def test_median(self):
        result = weighted_quantile([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 0.5)
        assert result == pytest.approx(3.0)

    def test_extreme_quantiles(self):
        vals = [10, 20, 30]
        w = [1, 1, 1]
        assert weighted_quantile(vals, w, 0.0) == pytest.approx(10.0)


class TestReplicateWeights:
    def test_replicate_weight_columns_sorted(self):
        cols = ["PWGTP10", "PWGTP2", "PWGTP", "PWGTP1", "other"]
        assert replicate_weight_columns(cols) == ["PWGTP1", "PWGTP2", "PWGTP10"]

    def test_sdr_standard_error(self):
        se = sdr_standard_error(10.0, [9.0, 11.0])
        expected = np.sqrt((4 / 2) * ((9 - 10) ** 2 + (11 - 10) ** 2))
        assert se == pytest.approx(expected)

    def test_confidence_interval(self):
        low, high = confidence_interval(10.0, 2.0, level=0.90)
        assert low == pytest.approx(10.0 - 1.645 * 2.0)
        assert high == pytest.approx(10.0 + 1.645 * 2.0)
