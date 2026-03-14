"""Canonical table schemas for standardized data.

Defines the target columns for each standardized table.
Source-specific standardizers must produce DataFrames matching these schemas.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# person_year_core — ACS-style annual analysis
# ---------------------------------------------------------------------------

PERSON_YEAR_CORE_COLUMNS = [
    "person_id",
    "household_id",
    "data_source",
    "survey_year",
    "calendar_year",
    "female",
    "age",
    "age_sq",
    "race_ethnicity",
    "education_level",
    "marital_status",
    "number_children",
    "children_under_5",
    "occupation_code",
    "industry_code",
    "class_of_worker",
    "self_employed",
    "weeks_worked",
    "usual_hours_week",
    "annual_hours",
    "work_from_home",
    "commute_minutes_one_way",
    "commute_mode",
    "state_fips",
    "residence_puma",
    "place_of_work_state",
    "place_of_work_puma",
    "hourly_wage_real",
    "annual_earnings_real",
    "wage_salary_income_real",
    "person_weight",
]

# ---------------------------------------------------------------------------
# person_month_core — CPS/SIPP-style monthly analysis
# ---------------------------------------------------------------------------

PERSON_MONTH_CORE_COLUMNS = [
    "person_id",
    "calendar_year",
    "month",
    "female",
    "employed",
    "labor_force_status",
    "usual_hours_week",
    "actual_hours_last_week",
    "paid_hourly",
    "hourly_wage_real",
    "weekly_earnings_real",
    "overtime_indicator",
    "multiple_jobholder",
    "occupation_code",
    "industry_code",
    "state_fips",
    "person_weight",
]

# ---------------------------------------------------------------------------
# person_day_timeuse — ATUS mechanism table
# ---------------------------------------------------------------------------

PERSON_DAY_TIMEUSE_COLUMNS = [
    "person_id",
    "calendar_year",
    "diary_date",
    "female",
    "employed",
    "minutes_paid_work_diary",
    "minutes_work_at_home_diary",
    "minutes_commute_related_travel",
    "minutes_housework",
    "minutes_childcare",
    "minutes_eldercare",
    "minutes_with_children",
    "person_weight",
]

# ---------------------------------------------------------------------------
# context_area_time — merged contextual controls
# ---------------------------------------------------------------------------

CONTEXT_AREA_TIME_COLUMNS = [
    "geography_level",
    "geography_key",
    "calendar_year",
    "local_unemployment_rate",
    "local_labor_force",
    "local_industry_avg_weekly_wage",
    "local_industry_employment",
    "local_occupation_avg_wage",
    "local_price_parity",
]
