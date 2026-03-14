from pathlib import Path

import pandas as pd
import pytest

from scripts.build_sce_public_analysis import extract_public_series, summarize_public_series, write_report


def test_extract_public_series_builds_gap_columns(tmp_path: Path):
    rows = [
        ["meta"] * 7,
        ["meta"] * 7,
        ["meta"] * 7,
        ["meta"] * 7,
        ["meta"] * 7,
        ["Date", "exp_mean", "exp_w", "exp_m", "rw_mean", "rw_w", "rw_m"],
        [None] * 7,
        ["Mar 2025", 66.7, 56.5, 76.4, 74.2, 65.6, 82.3],
        ["Jul 2025", 66.6, 60.5, 72.1, 82.5, 74.4, 90.1],
    ]
    path = tmp_path / "chart.xlsx"
    pd.DataFrame(rows).to_excel(path, sheet_name="Data", header=False, index=False)

    # Re-map expected live columns into the same positional slots used by the parser.
    raw = pd.read_excel(path, sheet_name="Data", header=None)
    full = pd.DataFrame(index=raw.index, columns=range(194))
    full.iloc[:, 0] = raw.iloc[:, 0]
    full.iloc[:, 176] = raw.iloc[:, 1]
    full.iloc[:, 182] = raw.iloc[:, 2]
    full.iloc[:, 183] = raw.iloc[:, 3]
    full.iloc[:, 186] = raw.iloc[:, 4]
    full.iloc[:, 192] = raw.iloc[:, 5]
    full.iloc[:, 193] = raw.iloc[:, 6]
    full.to_excel(path, sheet_name="Data", header=False, index=False)

    out = extract_public_series(path)
    assert list(out["reservation_wage_gap_men_minus_women"]) == pytest.approx([16.7, 15.7])
    assert list(out["expected_offer_wage_gap_men_minus_women"]) == pytest.approx([19.9, 11.6])


def test_write_report_mentions_reservation_wage(tmp_path: Path):
    series = pd.DataFrame({
        "date": pd.to_datetime(["2025-03-01", "2025-07-01"]),
        "expected_offer_wage_women": [56.5, 60.5],
        "expected_offer_wage_men": [76.4, 72.1],
        "reservation_wage_women": [65.6, 74.4],
        "reservation_wage_men": [82.3, 90.1],
    })
    summary = pd.DataFrame([
        {"metric": "latest_date", "value": "2025-07"},
        {"metric": "latest_expected_offer_wage_women", "value": 60.5},
        {"metric": "latest_expected_offer_wage_men", "value": 72.1},
        {"metric": "latest_expected_offer_wage_gap_men_minus_women", "value": 11.6},
        {"metric": "latest_reservation_wage_women", "value": 74.4},
        {"metric": "latest_reservation_wage_men", "value": 90.1},
        {"metric": "latest_reservation_wage_gap_men_minus_women", "value": 15.7},
        {"metric": "trailing_year_expected_offer_gap_mean", "value": 15.75},
        {"metric": "trailing_year_reservation_gap_mean", "value": 16.2},
        {"metric": "reservation_wage_gap_change_since_start", "value": -1.0},
    ])
    output = tmp_path / "sce_public_analysis.md"
    write_report(series, summary, output)
    text = output.read_text()
    assert "expected offer wage" in text.lower()
    assert "reservation wage" in text.lower()
    assert "supporting evidence" in text.lower()
