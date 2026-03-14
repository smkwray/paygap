from pathlib import Path

from scripts.build_sce_supplement import build_sce_measure_map, write_sce_report


def test_build_sce_measure_map_contains_reservation_wage():
    df = build_sce_measure_map()

    assert "reservation_wage" in set(df["construct"])
    assert not df["mergeable_into_acs_cps"].any()


def test_write_sce_report_marks_sce_as_supplement(tmp_path: Path):
    output = tmp_path / "sce_supplement.md"
    metrics = {
        "acs_2023_m5_pct_gap": 13.23,
        "acs_m5_mean_pct_gap": 13.77,
        "acs_s2_expected_gap_pct": 38.48,
        "acs_s2_ipw_gap_pct": 19.73,
        "cps_s2_expected_gap_pct": 31.99,
        "cps_s2_ipw_gap_pct": 19.25,
        "atus_housework_gap_minutes": 31.91,
        "atus_childcare_gap_minutes": 11.37,
        "atus_paid_work_gap_minutes": -67.96,
    }

    write_sce_report(metrics, build_sce_measure_map(), output)

    text = output.read_text()
    assert "supporting evidence" in text.lower()
    assert "Do not treat SCE as a drop-in control variable" in text
    assert "Reservation Wage" in text
