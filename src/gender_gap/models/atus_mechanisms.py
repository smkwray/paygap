"""ATUS mechanism summaries keyed to reproductive-stage groups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gender_gap.features.reproductive import add_reproductive_features
from gender_gap.utils.io import read_parquet

TIME_USE_COLUMNS = [
    "minutes_paid_work_diary",
    "minutes_housework",
    "minutes_childcare",
    "minutes_commute_related_travel",
    "minutes_eldercare",
    "minutes_work_at_home_diary",
]


def build_atus_mechanism_table(
    processed_path,
    respondent_path,
    roster_path,
) -> pd.DataFrame:
    """Build weighted ATUS mechanism summaries using respondent and roster family context."""
    timeuse = read_parquet(processed_path)
    respondent = read_parquet(respondent_path)
    roster = read_parquet(roster_path)

    enriched = enrich_atus_with_reproductive_features(timeuse, respondent, roster)
    rows = []

    for metric in [col for col in TIME_USE_COLUMNS if col in enriched.columns]:
        rows.append(_summary_row(enriched, metric, stage="overall"))
        for stage, sdf in enriched.groupby("reproductive_stage", dropna=False):
            rows.append(_summary_row(sdf, metric, stage=str(stage)))

    output = pd.DataFrame(rows)
    output.insert(0, "status", "ok")
    return output.sort_values(["metric", "reproductive_stage"], ignore_index=True)


def enrich_atus_with_reproductive_features(
    timeuse: pd.DataFrame,
    respondent: pd.DataFrame,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    """Merge raw ATUS family context onto the standardized time-use table."""
    base = timeuse.copy()
    respondent = respondent.copy()
    respondent["person_id"] = respondent["TUCASEID"].astype(str)

    resp_cols = [
        "person_id",
        "TUYEAR",
        "TELFS",
        "TRSPPRES",
        "TRTUNMPART",
        "TRTSPONLY",
        "TRTSPOUSE",
        "TRHHCHILD",
        "TRNHHCHILD",
        "TROHHCHILD",
        "TRCHILDNUM",
        "TRYHHCHILD",
        "TUFNWGTP",
        "TU20FWGT",
        "TEAGE",
        "TESEX",
    ]
    available_resp = [col for col in resp_cols if col in respondent.columns]
    merged = base.merge(
        respondent[available_resp],
        on="person_id",
        how="left",
        suffixes=("", "_resp"),
    )

    roster_features = _build_roster_features(roster)
    merged = merged.merge(roster_features, on="person_id", how="left")

    age_source = merged["TEAGE"] if "TEAGE" in merged.columns else merged.get(
        "respondent_age",
        pd.Series(np.nan, index=merged.index),
    )
    female_source = (
        merged["female"]
        if "female" in merged.columns
        else pd.Series(np.nan, index=merged.index)
    )
    sex_source = (
        merged["TESEX"] if "TESEX" in merged.columns else pd.Series(np.nan, index=merged.index)
    )
    employed_source = (
        merged["employed"]
        if "employed" in merged.columns
        else pd.Series(np.nan, index=merged.index)
    )
    telfs_source = (
        merged["TELFS"] if "TELFS" in merged.columns else pd.Series(np.nan, index=merged.index)
    )

    merged["age"] = pd.to_numeric(age_source, errors="coerce")
    merged["female"] = pd.to_numeric(female_source, errors="coerce").fillna(
        pd.to_numeric(sex_source, errors="coerce").eq(2).astype(float)
    )
    merged["employed"] = pd.to_numeric(employed_source, errors="coerce").fillna(
        pd.to_numeric(telfs_source, errors="coerce").isin([1, 2]).astype(float)
    )

    nonhh_children = pd.to_numeric(merged.get("TRNHHCHILD"), errors="coerce").fillna(0)
    own_hh_children = pd.to_numeric(merged.get("TROHHCHILD"), errors="coerce").fillna(0)
    own_children_total = pd.to_numeric(merged.get("own_child_count"), errors="coerce").fillna(0)
    own_children_total = own_children_total.where(
        merged.get("own_child_count").notna(),
        own_hh_children + nonhh_children,
    )

    merged["number_children"] = own_children_total
    merged["has_own_child"] = (own_children_total > 0).astype(int)
    merged["own_child_under6"] = pd.to_numeric(
        merged.get("own_child_under6"),
        errors="coerce",
    ).fillna(0).astype(int)
    merged["own_child_6_17_only"] = (
        merged["has_own_child"].eq(1) & merged["own_child_under6"].eq(0)
    ).astype(int)

    youngest_own_child = pd.to_numeric(merged.get("youngest_own_child_age"), errors="coerce")
    merged["recent_birth"] = (
        merged["female"].eq(1) & youngest_own_child.notna() & youngest_own_child.le(0)
    ).astype(int)
    merged["recent_marriage"] = 0

    partnered = (
        pd.to_numeric(merged.get("TRTSPOUSE"), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(merged.get("TRTSPONLY"), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(merged.get("TRTUNMPART"), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(merged.get("partner_present"), errors="coerce").fillna(0).gt(0)
    )
    merged["partner"] = partnered.astype(int)

    couple_type = pd.Series("unpartnered", index=merged.index, dtype="string")
    opposite = partnered & pd.to_numeric(
        merged.get("same_sex_couple_household"),
        errors="coerce",
    ).fillna(0).eq(0)
    same = partnered & pd.to_numeric(
        merged.get("same_sex_couple_household"),
        errors="coerce",
    ).fillna(0).eq(1)
    couple_type.loc[opposite] = "opposite_sex"
    couple_type.loc[same] = "same_sex"
    merged["opposite_sex_couple_household"] = opposite.astype(int)
    merged["couple_type"] = couple_type

    current_weight = pd.to_numeric(
        merged["person_weight"]
        if "person_weight" in merged.columns
        else pd.Series(np.nan, index=merged.index),
        errors="coerce",
    )
    covid_weight = pd.to_numeric(
        merged["TU20FWGT"]
        if "TU20FWGT" in merged.columns
        else pd.Series(np.nan, index=merged.index),
        errors="coerce",
    )
    final_weight = pd.to_numeric(
        merged["TUFNWGTP"]
        if "TUFNWGTP" in merged.columns
        else pd.Series(np.nan, index=merged.index),
        errors="coerce",
    )
    merged["person_weight"] = (
        current_weight.fillna(covid_weight).fillna(final_weight).fillna(1.0)
    )
    merged["annual_earnings_real"] = pd.Series(np.nan, index=merged.index, dtype=float)

    enriched = add_reproductive_features(merged)
    keep = [
        "person_id",
        "calendar_year",
        "female",
        "employed",
        "age",
        "person_weight",
        "number_children",
        "has_own_child",
        "own_child_under6",
        "own_child_6_17_only",
        "same_sex_couple_household",
        "opposite_sex_couple_household",
        "couple_type",
        "recent_birth",
        "recent_marriage",
        "reproductive_stage",
        *[col for col in TIME_USE_COLUMNS if col in enriched.columns],
    ]
    return enriched.loc[:, keep].copy()


def _build_roster_features(roster: pd.DataFrame) -> pd.DataFrame:
    frame = roster.copy()
    frame["person_id"] = frame["TUCASEID"].astype(str)
    rel = pd.to_numeric(frame["TERRP"], errors="coerce")
    age = pd.to_numeric(frame["TEAGE"], errors="coerce")
    sex = pd.to_numeric(frame["TESEX"], errors="coerce")

    own_child = rel.isin([22, 40])
    partner = rel.isin([20, 21])
    self_row = rel.isin([18, 19])

    def _min_age(mask: pd.Series) -> pd.Series:
        values = age.where(mask)
        return values.groupby(frame["person_id"]).min()

    own_count = own_child.groupby(frame["person_id"]).sum(min_count=1).fillna(0)
    under6 = (own_child & age.lt(6)).groupby(frame["person_id"]).sum(min_count=1).fillna(0)
    partner_present = partner.groupby(frame["person_id"]).sum(min_count=1).fillna(0)
    self_sex = sex.where(self_row).groupby(frame["person_id"]).first()
    self_age = age.where(self_row).groupby(frame["person_id"]).first()
    partner_sex = sex.where(partner).groupby(frame["person_id"]).first()

    result = pd.DataFrame(
        {
            "person_id": own_count.index.astype(str),
            "own_child_count": own_count.astype(float),
            "own_child_under6": under6.gt(0).astype(int),
            "youngest_own_child_age": _min_age(own_child),
            "partner_present": partner_present.gt(0).astype(int),
            "respondent_age": self_age,
            "same_sex_couple_household": (
                partner_present.gt(0)
                & self_sex.notna()
                & partner_sex.notna()
                & self_sex.eq(partner_sex)
            ).astype(int),
        }
    )
    return result.reset_index(drop=True)


def _summary_row(df: pd.DataFrame, metric: str, stage: str) -> dict:
    female = pd.to_numeric(df.get("female"), errors="coerce")
    value = pd.to_numeric(df.get(metric), errors="coerce")
    weight = pd.to_numeric(df.get("person_weight"), errors="coerce").fillna(1.0)
    male_mask = female.eq(0) & value.notna() & weight.gt(0)
    female_mask = female.eq(1) & value.notna() & weight.gt(0)
    male_mean = _weighted_mean(value.loc[male_mask], weight.loc[male_mask])
    female_mean = _weighted_mean(value.loc[female_mask], weight.loc[female_mask])
    return {
        "reproductive_stage": stage,
        "metric": metric,
        "male_mean_minutes": male_mean,
        "female_mean_minutes": female_mean,
        "gap_minutes": (
            female_mean - male_mean
            if pd.notna(male_mean) and pd.notna(female_mean)
            else np.nan
        ),
        "n_male": int(male_mask.sum()),
        "n_female": int(female_mask.sum()),
        "weighted_n_male": float(weight.loc[male_mask].sum()),
        "weighted_n_female": float(weight.loc[female_mask].sum()),
    }


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(np.average(values.astype(float), weights=weights.astype(float)))
