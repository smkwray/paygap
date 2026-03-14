from __future__ import annotations

import pandas as pd

from scripts.build_sipp_models import SIPP_BLOCKS, _pct_from_log_coef, build_report
from gender_gap.models.ols import results_to_dataframe, run_sequential_ols


def _model_df() -> pd.DataFrame:
    rows = []
    for month in range(1, 7):
        rows.append(
            {
                "female": 0,
                "month": month,
                "occupation_code": "1110" if month % 2 else "2210",
                "industry_code": "3110" if month % 2 else "5410",
                "usual_hours_week": 40,
                "paid_hourly": 1,
                "multiple_jobholder": 0,
                "hourly_wage_real": 32 + month,
                "person_weight": 1,
                "employed": 1,
            }
        )
        rows.append(
            {
                "female": 1,
                "month": month,
                "occupation_code": "1110" if month % 2 else "2210",
                "industry_code": "3110" if month % 2 else "5410",
                "usual_hours_week": 38,
                "paid_hourly": 1 if month % 2 else 0,
                "multiple_jobholder": 0 if month % 2 else 1,
                "hourly_wage_real": 24 + month,
                "person_weight": 1,
                "employed": 1,
            }
        )
    return pd.DataFrame(rows)


def test_sipp_blocks_run():
    results = run_sequential_ols(
        _model_df(),
        outcome="log_hourly_wage_real",
        weight_col="person_weight",
        blocks=SIPP_BLOCKS,
    )
    df = results_to_dataframe(results)
    assert df["model"].tolist() == ["SIPP0", "SIPP1", "SIPP2", "SIPP3"]
    assert (df["female_coef"] < 0).all()


def test_pct_from_log_coef_negative():
    assert _pct_from_log_coef(-0.1) < 0


def test_build_report_mentions_sipp3(tmp_path):
    ols_df = pd.DataFrame(
        {
            "model": ["SIPP0", "SIPP3"],
            "female_coef": [-0.2, -0.1],
            "pct_gap": [-18.13, -9.52],
            "r_squared": [0.05, 0.25],
            "n_obs": [100, 100],
        }
    )
    path = tmp_path / "sipp_models.md"
    build_report(ols_df, 15.0, path)
    text = path.read_text()
    assert "SIPP3" in text
    assert "Raw hourly wage gap" in text
    assert "adjusted-gap surface" in text
