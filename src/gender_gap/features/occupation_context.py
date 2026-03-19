"""O*NET occupational context feature builders."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gender_gap.crosswalks.occupation_crosswalks import (
    census_occ_to_soc_major,
    harmonize_occupation_codes,
    onet_soc_to_census_soc,
)
from gender_gap.utils.yaml_compat import load_yaml

logger = logging.getLogger(__name__)


def build_onet_indices(
    onet_dir: Path,
    recipe_path: Path,
    granularity: str = "major_group",
) -> pd.DataFrame:
    """Build seeded O*NET indices from the text database.

    Parameters
    ----------
    granularity : str
        ``"major_group"`` (default) aggregates to 2-digit SOC major group,
        matching existing behavior.  ``"detailed"`` keeps per-O*NET-SOC-code
        indices for finer occupation matching.
    """
    with open(recipe_path, encoding="utf-8") as f:
        recipe = load_yaml(f.read()) or {}

    work_context_name = recipe.get("sources", {}).get("work_context", "Work Context.txt")
    work_context_path = onet_dir / work_context_name
    if not work_context_path.exists():
        raise FileNotFoundError(f"Required O*NET file missing: {work_context_path}")

    raw = _read_onet_table(work_context_path)
    col_soc = _find_column(raw, ["O*NET-SOC Code", "ONET_SOC_CODE", "onet_soc_code"])
    col_element = _find_column(raw, ["Element Name", "ELEMENT_NAME", "element_name"])
    col_scale = _find_column(raw, ["Scale ID", "SCALE_ID", "scale_id"])
    col_value = _find_column(raw, ["Data Value", "DATA_VALUE", "data_value", "value"])

    standardized = raw[[col_soc, col_element, col_scale, col_value]].copy()
    standardized.columns = ["onet_soc_code", "element_name", "scale_id", "data_value"]
    standardized["onet_soc_code"] = standardized["onet_soc_code"].astype(str).str.strip()
    standardized["element_name"] = standardized["element_name"].astype(str).str.strip()
    standardized["scale_id"] = standardized["scale_id"].astype(str).str.strip()
    standardized["data_value"] = pd.to_numeric(standardized["data_value"], errors="coerce")
    standardized = standardized.dropna(subset=["onet_soc_code", "element_name", "data_value"])

    built: dict[str, pd.DataFrame] = {}
    for index_name, spec in (recipe.get("indices") or {}).items():
        if "composite" in spec:
            continue
        built[index_name] = _build_component_index(standardized, spec, index_name)

    index_df = None
    for index_name, frame in built.items():
        frame = frame.rename(columns={"index_value": index_name})
        if index_df is None:
            index_df = frame
        else:
            index_df = index_df.merge(frame, on=["onet_soc_code", "soc_major_group"], how="outer")

    if index_df is None:
        return pd.DataFrame(columns=["onet_soc_code", "soc_major_group"])

    for index_name, spec in (recipe.get("indices") or {}).items():
        if "composite" not in spec:
            continue
        numerator = 0.0
        weight_total = 0.0
        for component in spec["composite"]:
            src = component["index"]
            weight = float(component.get("weight", 1.0))
            if src in index_df.columns:
                numerator = numerator + (index_df[src] * weight)
                weight_total += abs(weight)
        if weight_total:
            index_df[index_name] = numerator / weight_total
        else:
            index_df[index_name] = pd.NA

    if granularity == "detailed":
        # Return per-O*NET-SOC-code indices.  Add a 6-digit SOC key
        # (strip the .XX suffix) for merging with Census SOC codes.
        index_df["soc_detailed"] = (
            index_df["onet_soc_code"].str.replace(r"\.\d+$", "", regex=True)
        )
        logger.info(
            "O*NET detailed indices: %d SOC codes, %d unique 6-digit SOC",
            len(index_df), index_df["soc_detailed"].nunique(),
        )
        return index_df.reset_index(drop=True)

    # Default: aggregate to 2-digit SOC major group
    value_cols = [
        col for col in index_df.columns if col not in {"onet_soc_code", "soc_major_group"}
    ]
    aggregated = (
        index_df.groupby("soc_major_group", as_index=False)[value_cols]
        .mean()
        .sort_values("soc_major_group")
        .reset_index(drop=True)
    )
    return aggregated


def merge_onet_context(
    df: pd.DataFrame,
    onet_indices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge O*NET context using a major-group bridge and report coverage."""
    out = df.copy()
    occ = pd.to_numeric(out.get("occupation_code"), errors="coerce")
    out["soc_major_group"] = census_occ_to_soc_major(occ)
    merged = out.merge(onet_indices, on="soc_major_group", how="left")

    if "job_rigidity" in merged.columns:
        ranked = merged["job_rigidity"].rank(method="first")
        valid = merged["job_rigidity"].notna()
        merged["job_rigidity_quartile"] = pd.Series(pd.NA, index=merged.index, dtype="string")
        if valid.sum() >= 4:
            merged.loc[valid, "job_rigidity_quartile"] = pd.qcut(
                ranked.loc[valid], 4, labels=["Q1", "Q2", "Q3", "Q4"]
            ).astype("string")

    coverage = build_onet_merge_coverage(merged)
    return merged, coverage


def merge_onet_context_detailed(
    df: pd.DataFrame,
    onet_indices_detailed: pd.DataFrame,
    onet_indices_major: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge O*NET context at detailed SOC level with major-group fallback.

    Parameters
    ----------
    df : DataFrame
        Standardized ACS frame with ``occupation_code`` and optionally
        ``survey_year``.
    onet_indices_detailed : DataFrame
        Output of ``build_onet_indices(granularity="detailed")``.  Must
        contain ``soc_detailed`` and the index columns.
    onet_indices_major : DataFrame or None
        Optional major-group indices for fallback.  If None, unmatched
        rows get NaN for all index columns.

    Returns
    -------
    (merged_df, coverage_df)
    """
    out = df.copy()
    occ = pd.to_numeric(out.get("occupation_code"), errors="coerce")
    survey_year = out.get("survey_year")

    # Map Census 4-digit codes to 6-digit SOC via the harmonized lookup
    harmonized = harmonize_occupation_codes(occ, survey_year=survey_year)
    # Extract SOC code and normalize to "XX-XXXX" format for matching
    soc_raw = harmonized.get("soc_major_group")  # fallback
    soc_detailed_col = harmonized.get("occupation_harmonized_code")

    # The harmonized lookup stores SOC codes in soc_code fields.
    # Try to use the full SOC code from the codebook.
    # The soc_code_2018 from the codebook is typically "XX-XXXX" format.
    # We need to extract it — it's embedded in the harmonized frame.
    # Reconstruct from the occupation_harmonized_code + known SOC ranges.
    out["soc_major_group"] = census_occ_to_soc_major(occ)

    # Build a detailed SOC key from the Census occupation code.
    # O*NET soc_detailed is "XX-XXXX", Census occupation codes are 4-digit.
    # The harmonized crosswalk maps Census codes to SOC, but we need the
    # 6-digit SOC.  Use the O*NET data itself to build the bridge:
    # aggregate O*NET to unique soc_detailed per soc_major_group, then
    # find the closest match within each major group.
    #
    # Simpler approach: average O*NET indices by soc_detailed (removing
    # the .XX specialty suffix), then merge on soc_major_group + pick
    # the closest match.  But that requires a proper crosswalk.
    #
    # Most practical approach: merge on soc_major_group first, then
    # within each major group, use the detailed indices to get
    # occupation-specific values where a match exists.

    # Step 1: Build a Census-occ → soc_detailed bridge from the
    # harmonized occupation codebook
    _build_census_soc_bridge(out, harmonized, onet_indices_detailed)

    # Step 2: Merge detailed indices on soc_detailed
    index_cols = [
        col for col in onet_indices_detailed.columns
        if col not in {"onet_soc_code", "soc_major_group", "soc_detailed"}
    ]
    # Average O*NET across the .XX suffixes to get one row per soc_detailed
    detailed_agg = (
        onet_indices_detailed.groupby("soc_detailed", as_index=False)[index_cols]
        .mean()
    )

    merged = out.merge(detailed_agg, on="soc_detailed", how="left")

    # Step 3: Fallback to major-group for unmatched rows
    if onet_indices_major is not None:
        unmatched = merged[index_cols[0]].isna() if index_cols else pd.Series(False, index=merged.index)
        if unmatched.any():
            fallback = merged.loc[unmatched].drop(columns=index_cols, errors="ignore")
            fallback = fallback.merge(onet_indices_major, on="soc_major_group", how="left")
            for col in index_cols:
                if col in fallback.columns:
                    merged.loc[unmatched, col] = fallback[col].values

    # Quartile job_rigidity
    if "job_rigidity" in merged.columns:
        ranked = merged["job_rigidity"].rank(method="first")
        valid = merged["job_rigidity"].notna()
        merged["job_rigidity_quartile"] = pd.Series(pd.NA, index=merged.index, dtype="string")
        if valid.sum() >= 4:
            merged.loc[valid, "job_rigidity_quartile"] = pd.qcut(
                ranked.loc[valid], 4, labels=["Q1", "Q2", "Q3", "Q4"]
            ).astype("string")

    coverage = build_onet_merge_coverage(merged, merge_key="soc_detailed")
    return merged, coverage


def _build_census_soc_bridge(
    df: pd.DataFrame,
    harmonized: pd.DataFrame,
    onet_detailed: pd.DataFrame,
) -> None:
    """Add a ``soc_detailed`` column to df by bridging Census codes to O*NET SOC.

    Uses the harmonized occupation lookup's ``soc_code_detailed`` column
    (built from the Census codebook's actual SOC codes) to map each Census
    4-digit occupation code to a 6-digit SOC code.  Only matches codes that
    exist in the O*NET data; unmatched codes get NaN and will fall back to
    major-group averages in the caller.
    """
    available_soc = set(onet_detailed["soc_detailed"].dropna().unique())

    # The harmonized frame (from harmonize_occupation_codes) has
    # soc_code_detailed per row — use it directly.
    if "soc_code_detailed" in harmonized.columns:
        raw_soc = harmonized["soc_code_detailed"].astype(str).str.strip()
        # Only keep SOC codes that actually exist in the O*NET data
        matched = raw_soc.where(raw_soc.isin(available_soc))
        df["soc_detailed"] = matched.values
        n_matched = matched.notna().sum()
        n_total = len(df)
        logger.info(
            "Census→SOC detailed bridge: %d/%d rows matched (%.1f%%)",
            n_matched, n_total, 100 * n_matched / max(n_total, 1),
        )
    else:
        logger.warning(
            "soc_code_detailed not in harmonized lookup; "
            "rebuild the crosswalk CSV from the Census codebook to enable detailed matching"
        )
        df["soc_detailed"] = pd.NA


def build_onet_merge_coverage(
    df: pd.DataFrame,
    merge_key: str = "soc_major_group",
) -> pd.DataFrame:
    """Summarize O*NET coverage by survey year."""
    if "job_rigidity" not in df.columns:
        return pd.DataFrame(columns=["survey_year", "n_obs", "n_matched", "match_rate"])

    rows = []
    group_cols = ["survey_year"] if "survey_year" in df.columns else []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]
    for key, gdf in grouped:
        rows.append(
            {
                "survey_year": key if key is not None else "pooled",
                "n_obs": int(len(gdf)),
                "n_matched": int(gdf["job_rigidity"].notna().sum()),
                "match_rate": float(gdf["job_rigidity"].notna().mean()) if len(gdf) else 0.0,
                "merge_key": merge_key,
            }
        )
    return pd.DataFrame(rows)


def _build_component_index(
    standardized: pd.DataFrame,
    spec: dict,
    index_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for component in spec.get("components", []):
        mask = standardized["element_name"].str.casefold() == component["element"].casefold()
        if "scale_id" in spec:
            mask &= standardized["scale_id"].str.casefold() == str(spec["scale_id"]).casefold()
        component_df = standardized.loc[mask, ["onet_soc_code", "data_value"]].copy()
        if component_df.empty:
            logger.warning("O*NET component missing for %s: %s", index_name, component["element"])
            continue
        component_df["weighted_value"] = component_df["data_value"] * float(
            component.get("weight", 1.0)
        )
        component_df["weight"] = abs(float(component.get("weight", 1.0)))
        frames.append(component_df)

    if not frames:
        return pd.DataFrame(columns=["onet_soc_code", "soc_major_group", "index_value"])

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby("onet_soc_code", as_index=False)
        .agg(weighted_value=("weighted_value", "sum"), weight=("weight", "sum"))
    )
    grouped["index_value"] = grouped["weighted_value"] / grouped["weight"]
    grouped["soc_major_group"] = onet_soc_to_census_soc(grouped["onet_soc_code"])
    return grouped[["onet_soc_code", "soc_major_group", "index_value"]]


def _read_onet_table(path: Path) -> pd.DataFrame:
    for sep in ("\t", ","):
        df = pd.read_csv(path, sep=sep, engine="python")
        if df.shape[1] > 1:
            return df
    return pd.read_csv(path)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lowered = {str(col).casefold(): col for col in df.columns}
    for candidate in candidates:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]
    raise KeyError(f"Could not find any of {candidates} in {list(df.columns)}")
