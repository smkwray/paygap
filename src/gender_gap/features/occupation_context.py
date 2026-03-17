"""O*NET occupational context feature builders."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gender_gap.crosswalks.occupation_crosswalks import (
    census_occ_to_soc_major,
    onet_soc_to_census_soc,
)
from gender_gap.utils.yaml_compat import load_yaml

logger = logging.getLogger(__name__)


def build_onet_indices(onet_dir: Path, recipe_path: Path) -> pd.DataFrame:
    """Build seeded O*NET indices from the text database."""
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


def build_onet_merge_coverage(df: pd.DataFrame) -> pd.DataFrame:
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
                "merge_key": "soc_major_group",
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
