"""Tests for O*NET occupational context features."""

from pathlib import Path

import pandas as pd
import pytest

from gender_gap.features.occupation_context import build_onet_indices, merge_onet_context


def test_build_onet_indices_requires_work_context(tmp_path: Path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "sources:\n  work_context: Work Context.txt\nindices:\n  autonomy:\n    source: work_context\n    scale_id: CX\n    components:\n      - element: Freedom to Make Decisions\n        weight: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        build_onet_indices(tmp_path, recipe)


def test_build_onet_indices_and_merge(tmp_path: Path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """
sources:
  work_context: "Work Context.txt"
indices:
  autonomy:
    source: work_context
    scale_id: "CX"
    components:
      - element: "Freedom to Make Decisions"
        weight: 1.0
  time_pressure:
    source: work_context
    scale_id: "CX"
    components:
      - element: "Time Pressure"
        weight: 1.0
  job_rigidity:
    composite:
      - index: "time_pressure"
        weight: 0.5
      - index: "autonomy"
        weight: -0.5
""",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "O*NET-SOC Code": ["11-1011.00", "11-1011.00", "29-1141.00", "29-1141.00"],
            "Element Name": ["Freedom to Make Decisions", "Time Pressure", "Freedom to Make Decisions", "Time Pressure"],
            "Scale ID": ["CX", "CX", "CX", "CX"],
            "Data Value": [80, 40, 50, 70],
        }
    ).to_csv(tmp_path / "Work Context.txt", sep="\t", index=False)

    onet = build_onet_indices(tmp_path, recipe)
    assert {"autonomy", "time_pressure", "job_rigidity", "soc_major_group"}.issubset(onet.columns)

    df = pd.DataFrame({"occupation_code": [100, 3050], "survey_year": [2023, 2023]})
    merged, coverage = merge_onet_context(df, onet)
    assert merged["job_rigidity"].notna().all()
    assert coverage["n_matched"].iloc[0] == 2


def test_build_onet_indices_supports_ct_schedule_and_work_outcomes_name(tmp_path: Path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """
sources:
  work_context: "Work Context.txt"
indices:
  autonomy:
    source: work_context
    scale_id: "CX"
    components:
      - element: "Freedom to Make Decisions"
        weight: 1.0
  schedule_unpredictability:
    source: work_context
    scale_id: "CT"
    components:
      - element: "Work Schedules"
        weight: 1.0
      - element: "Duration of Typical Work Week"
        weight: 0.5
  coordination_responsibility:
    source: work_context
    scale_id: "CX"
    components:
      - element: "Coordinate or Lead Others in Accomplishing Work Activities"
        weight: 1.0
      - element: "Frequency of Decision Making"
        weight: 1.0
      - element: "Work Outcomes and Results of Other Workers"
        weight: 0.75
  job_rigidity:
    composite:
      - index: "schedule_unpredictability"
        weight: 0.5
      - index: "autonomy"
        weight: -0.5
""",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "O*NET-SOC Code": [
                "11-1011.00",
                "11-1011.00",
                "11-1011.00",
                "11-1011.00",
                "11-1011.00",
            ],
            "Element Name": [
                "Freedom to Make Decisions",
                "Work Schedules",
                "Duration of Typical Work Week",
                "Coordinate or Lead Others in Accomplishing Work Activities",
                "Work Outcomes and Results of Other Workers",
            ],
            "Scale ID": ["CX", "CT", "CT", "CX", "CX"],
            "Data Value": [80, 2, 3, 60, 50],
        }
    ).to_csv(tmp_path / "Work Context.txt", sep="\t", index=False)

    onet = build_onet_indices(tmp_path, recipe)

    assert onet["schedule_unpredictability"].notna().all()
    assert onet["coordination_responsibility"].notna().all()
    assert onet["job_rigidity"].notna().all()
