"""Survey weight utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_mean(values, weights) -> float:
    """Compute weighted mean, dropping NaN pairs."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def weighted_quantile(values, weights, quantile: float) -> float:
    """Compute a single weighted quantile."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    v, w = values[mask], weights[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumw = np.cumsum(w)
    cutoff = quantile * cumw[-1]
    idx = np.searchsorted(cumw, cutoff)
    return float(v[min(idx, len(v) - 1)])


def replicate_weight_columns(
    columns,
    prefix: str = "PWGTP",
    main_weight: str = "PWGTP",
) -> list[str]:
    """Return replicate-weight columns sorted by numeric suffix."""
    repweights = [
        str(col) for col in columns
        if str(col).startswith(prefix) and str(col) != main_weight
    ]
    return sorted(repweights, key=lambda col: int(col.replace(prefix, "")))


def sdr_variance(
    full_estimate: float,
    replicate_estimates,
    scale: float | None = None,
) -> float:
    """Compute ACS SDR variance from replicate estimates.

    ACS PUMS uses 80 replicate estimates with a factor of 4/80.
    """
    reps = np.asarray(replicate_estimates, dtype=float)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0 or not np.isfinite(full_estimate):
        return np.nan
    if scale is None:
        scale = 4.0 / reps.size
    diffs = reps - float(full_estimate)
    return float(scale * np.sum(diffs ** 2))


def sdr_standard_error(
    full_estimate: float,
    replicate_estimates,
    scale: float | None = None,
) -> float:
    """Compute ACS SDR standard error."""
    variance = sdr_variance(full_estimate, replicate_estimates, scale=scale)
    return float(np.sqrt(variance)) if np.isfinite(variance) else np.nan


def confidence_interval(
    estimate: float,
    standard_error: float,
    level: float = 0.90,
) -> tuple[float, float]:
    """Build a symmetric confidence interval from an estimate and SE."""
    if not np.isfinite(estimate) or not np.isfinite(standard_error):
        return (np.nan, np.nan)
    z_map = {
        0.90: 1.645,
        0.95: 1.96,
    }
    z = z_map.get(level)
    if z is None:
        raise ValueError("Supported levels are 0.90 and 0.95")
    margin = z * standard_error
    return (float(estimate - margin), float(estimate + margin))


def sdr_summary(
    estimate: float,
    replicate_estimates,
) -> dict[str, float]:
    """Return a compact ACS SDR summary."""
    se = sdr_standard_error(estimate, replicate_estimates)
    ci90_low, ci90_high = confidence_interval(estimate, se, level=0.90)
    ci95_low, ci95_high = confidence_interval(estimate, se, level=0.95)
    return {
        "estimate": float(estimate),
        "se": se,
        "moe90": 1.645 * se if np.isfinite(se) else np.nan,
        "ci90_low": ci90_low,
        "ci90_high": ci90_high,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }
