"""SIPP reproductive-burden robustness surface."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gender_gap.features.reproductive import add_repro_interactions, add_reproductive_features
from gender_gap.models.ols import coefficient_table, results_to_dataframe, run_sequential_ols

SIPP_REPRO_BLOCKS = {
    "SIPP0": ["female"],
    "SIPP1": ["female", "age", "age_sq", "C(month)"],
    "SIPP2": [
        "female",
        "age",
        "age_sq",
        "C(month)",
        "C(occupation_code)",
        "C(industry_code)",
        "usual_hours_week",
        "paid_hourly",
        "multiple_jobholder",
    ],
    "SIPP3_reproductive": [
        "female",
        "age",
        "age_sq",
        "C(month)",
        "C(occupation_code)",
        "C(industry_code)",
        "usual_hours_week",
        "paid_hourly",
        "multiple_jobholder",
        "has_own_child",
        "own_child_under6",
        "own_child_6_17_only",
        "C(couple_type)",
        "C(reproductive_stage)",
        "age_first_birth",
    ],
}


def build_sipp_robustness_table(
    standardized_path,
    raw_path,
    survey_year: int = 2023,
) -> pd.DataFrame:
    """Build a lightweight SIPP robustness surface with reproductive controls."""
    std = pd.read_parquet(standardized_path)
    raw = _read_raw_subset(raw_path)
    df = _merge_reproductive_slice(std, raw, survey_year=survey_year)
    sample = _analysis_sample(df)

    if sample.empty:
        return pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "section": "status",
                    "metric": "empty_sample",
                    "value": np.nan,
                    "note": (
                        "No employed SIPP worker-month sample with positive hourly wage "
                        "after merge."
                    ),
                }
            ]
        )

    ols = run_sequential_ols(
        sample,
        outcome="log_hourly_wage_real",
        weight_col="person_weight",
        blocks=SIPP_REPRO_BLOCKS,
    )
    ols_df = results_to_dataframe(ols)
    rows = []
    for row in ols_df.itertuples(index=False):
        rows.append(
            {
                "status": "ok_partial",
                "section": "ols",
                "metric": row.model,
                "value": row.female_coef,
                "note": (
                    f"pct_gap={(np.exp(row.female_coef)-1.0)*100.0:.2f}; "
                    f"r2={row.r_squared:.4f}; n={int(row.n_obs)}"
                ),
            }
        )

    stage_gap = _stage_gap_rows(sample)
    rows.extend(stage_gap)
    rows.append(
        {
            "status": "ok_partial",
            "section": "status",
            "metric": "fertility_risk_proxy",
            "value": np.nan,
            "note": (
                "Public-use SIPP supports reproductive controls and age-at-first-birth, "
                "but not a clean ACS-style recent-birth/fertility-risk score."
            ),
        }
    )
    rows.append(
        {
            "status": "ok_partial",
            "section": "status",
            "metric": "sample_size",
            "value": float(len(sample)),
            "note": (
                "Employed worker-month sample with positive hourly wage and "
                "merged reproductive fields."
            ),
        }
    )
    return pd.DataFrame(rows)


def _read_raw_subset(raw_path) -> pd.DataFrame:
    cols = [
        "SSUID",
        "PNUM",
        "MONTHCODE",
        "TAGE",
        "ESEX",
        "EPNSPOUSE",
        "EPNSPOUS_EHC",
        "RPAR1SEX_EHC",
        "RPAR2SEX_EHC",
        "TAGE_FB",
        "THHLDSTATUS",
    ] + [f"RPNCHILD{i}" for i in range(1, 12)]
    return pd.read_stata(raw_path, columns=cols, convert_categoricals=False)


def _merge_reproductive_slice(
    standardized: pd.DataFrame,
    raw: pd.DataFrame,
    survey_year: int,
) -> pd.DataFrame:
    base = standardized.copy()
    raw = raw.copy()
    raw["person_id"] = raw["SSUID"].astype("string") + "_" + raw["PNUM"].astype("Int64").astype(
        "string"
    )
    raw["month"] = pd.to_numeric(raw["MONTHCODE"], errors="coerce").astype("Int64")

    person_age = raw[["SSUID", "month", "PNUM", "TAGE"]].rename(
        columns={"PNUM": "child_pnum", "TAGE": "child_age"}
    )
    child_rows = []
    for i in range(1, 12):
        col = f"RPNCHILD{i}"
        tmp = raw[["SSUID", "month", "person_id", col]].rename(columns={col: "child_pnum"})
        tmp["child_slot"] = i
        child_rows.append(tmp)
    child_links = pd.concat(child_rows, ignore_index=True)
    child_links["child_pnum"] = pd.to_numeric(child_links["child_pnum"], errors="coerce")
    child_links = child_links.dropna(subset=["child_pnum"])
    child_links = child_links.merge(person_age, on=["SSUID", "month", "child_pnum"], how="left")

    child_summary = child_links.groupby("person_id", as_index=False).agg(
        own_child_count=("child_pnum", "count"),
        youngest_own_child_age=("child_age", "min"),
    )
    child_summary["own_child_under6"] = pd.to_numeric(
        child_summary["youngest_own_child_age"],
        errors="coerce",
    ).lt(6).fillna(False).astype(int)

    sex_lookup = raw[["SSUID", "month", "PNUM", "ESEX"]].rename(
        columns={"PNUM": "partner_pnum", "ESEX": "partner_sex"}
    )
    raw["partner_pnum"] = pd.to_numeric(raw["EPNSPOUS_EHC"], errors="coerce").fillna(
        pd.to_numeric(raw["EPNSPOUSE"], errors="coerce")
    )
    partnered = raw.merge(sex_lookup, on=["SSUID", "month", "partner_pnum"], how="left")
    partnered["same_sex_couple_household"] = (
        pd.to_numeric(partnered["partner_pnum"], errors="coerce").notna()
        & pd.to_numeric(partnered["partner_sex"], errors="coerce").eq(
            pd.to_numeric(partnered["ESEX"], errors="coerce")
        )
    ).astype(int)

    features = partnered[
        [
            "person_id",
            "month",
            "TAGE",
            "ESEX",
            "TAGE_FB",
            "same_sex_couple_household",
            "partner_pnum",
            "THHLDSTATUS",
        ]
    ].merge(child_summary, on="person_id", how="left")

    features["age"] = pd.to_numeric(features["TAGE"], errors="coerce")
    features["female"] = pd.to_numeric(features["ESEX"], errors="coerce").eq(2).astype(int)
    features["age_first_birth"] = pd.to_numeric(features["TAGE_FB"], errors="coerce")
    features["partner"] = pd.to_numeric(features["partner_pnum"], errors="coerce").notna().astype(
        int
    )
    features["has_own_child"] = (
        pd.to_numeric(features["own_child_count"], errors="coerce").fillna(0).gt(0).astype(int)
    )
    features["own_child_under6"] = (
        pd.to_numeric(features["own_child_under6"], errors="coerce").fillna(0).astype(int)
    )
    features["own_child_6_17_only"] = (
        features["has_own_child"].eq(1) & features["own_child_under6"].eq(0)
    ).astype(int)
    features["recent_birth"] = 0
    features["recent_marriage"] = 0
    features["opposite_sex_couple_household"] = (
        features["partner"].eq(1) & features["same_sex_couple_household"].eq(0)
    ).astype(int)
    features["number_children"] = pd.to_numeric(
        features["own_child_count"],
        errors="coerce",
    ).fillna(0)
    features["children_under_5"] = features["own_child_under6"]

    merge_cols = [
        "person_id",
        "month",
        "age",
        "female",
        "age_first_birth",
        "partner",
        "same_sex_couple_household",
        "opposite_sex_couple_household",
        "number_children",
        "children_under_5",
        "has_own_child",
        "own_child_under6",
        "own_child_6_17_only",
        "recent_birth",
        "recent_marriage",
    ]
    merged = base.merge(
        features[merge_cols].drop_duplicates(["person_id", "month"]),
        on=["person_id", "month"],
        how="left",
    )
    merged["calendar_year"] = merged.get("calendar_year", survey_year)
    if "female_x" in merged.columns or "female_y" in merged.columns:
        merged["female"] = pd.to_numeric(
            merged.get("female_x", merged.get("female_y")),
            errors="coerce",
        ).fillna(pd.to_numeric(merged.get("female_y", merged.get("female_x")), errors="coerce"))
    if "age_x" in merged.columns or "age_y" in merged.columns:
        merged["age"] = pd.to_numeric(
            merged.get("age_x", merged.get("age_y")),
            errors="coerce",
        ).fillna(pd.to_numeric(merged.get("age_y", merged.get("age_x")), errors="coerce"))
    merged["age_sq"] = pd.to_numeric(merged["age"], errors="coerce") ** 2
    merged["hourly_wage_real"] = pd.to_numeric(merged["hourly_wage_real"], errors="coerce")
    merged["log_hourly_wage_real"] = np.log(
        merged["hourly_wage_real"].where(merged["hourly_wage_real"] > 0)
    )
    merged["usual_hours_week"] = pd.to_numeric(merged["usual_hours_week"], errors="coerce")
    merged["paid_hourly"] = pd.to_numeric(merged["paid_hourly"], errors="coerce")
    merged["multiple_jobholder"] = pd.to_numeric(merged["multiple_jobholder"], errors="coerce")
    merged["person_weight"] = pd.to_numeric(merged["person_weight"], errors="coerce")
    merged = add_reproductive_features(merged)
    merged = add_repro_interactions(merged)
    return merged


def _analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    age = pd.to_numeric(df.get("age"), errors="coerce")
    wage = pd.to_numeric(df.get("hourly_wage_real"), errors="coerce")
    employed = pd.to_numeric(df.get("employed"), errors="coerce")
    weight = pd.to_numeric(df.get("person_weight"), errors="coerce")
    mask = age.between(25, 54, inclusive="both") & employed.eq(1) & wage.gt(0) & weight.gt(0)
    return df.loc[mask].copy()


def _stage_gap_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for stage, sdf in df.groupby("reproductive_stage", dropna=False):
        table = coefficient_table(
            sdf,
            model_name="SIPP_stage_gap",
            outcome="log_hourly_wage_real",
            weight_col="person_weight",
            blocks={"SIPP_stage_gap": ["female", "C(month)"]},
        )
        if table.empty:
            continue
        female_row = table.loc[table["term"] == "female"]
        if female_row.empty:
            continue
        row = female_row.iloc[0]
        rows.append(
            {
                "status": "ok_partial",
                "section": "stage_gap",
                "metric": str(stage),
                "value": row["coef"],
                "note": (
                    f"pct_gap={(np.exp(row['coef'])-1.0)*100.0:.2f}; "
                    f"n={int(row['n_obs']) if 'n_obs' in row else len(sdf)}"
                ),
            }
        )
    return rows
