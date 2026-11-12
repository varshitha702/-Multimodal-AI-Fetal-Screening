"""Verify ml/intergrowth.py against the official INTERGROWTH-21st fetal HC z-score table."""
from pathlib import Path

import pandas as pd
import pytest

from ml.intergrowth import (
    assess_hc,
    centile_curves,
    hc_at_z,
    hc_mean_mm,
    hc_sd_mm,
    z_to_centile,
)

TABLE = Path(__file__).parent / "data" / "intergrowth21_fetal_hc_zscores.csv"


def table():
    return pd.read_csv(TABLE, comment="#")


def test_every_published_table_cell_is_reproduced():
    """27 gestational ages x 7 z-scores = 189 published values (rounded to 0.1 mm)."""
    t = table()
    assert len(t) == 27 and t.ga_weeks.min() == 14 and t.ga_weeks.max() == 40
    checked = 0
    for _, row in t.iterrows():
        for z in range(-3, 4):
            assert hc_at_z(row.ga_weeks, z) == pytest.approx(row[f"z_{z}"], abs=0.06), (row.ga_weeks, z)
            checked += 1
    assert checked == 189


@pytest.mark.parametrize("ga, median", [(14, 97.9), (20, 172.5), (30, 278.4), (40, 333.9)])
def test_published_medians(ga, median):
    assert round(hc_mean_mm(ga), 1) == median


def test_measurement_at_median_is_50th_centile():
    a = assess_hc(hc_mean_mm(24.0), 24.0)
    assert a.z_score == pytest.approx(0.0, abs=1e-9)
    assert a.centile == pytest.approx(50.0)
    assert a.within_p3_p97


def test_centile_from_z():
    assert z_to_centile(1.88) == pytest.approx(97.0, abs=0.1)
    assert z_to_centile(-1.88) == pytest.approx(3.0, abs=0.1)


def test_small_head_flagged_outside_p3_p97():
    ga = 30.0
    a = assess_hc(hc_at_z(ga, -2.5), ga)
    assert a.z_score == pytest.approx(-2.5)
    assert not a.within_p3_p97
    assert a.p3_mm < a.median_mm < a.p97_mm


@pytest.mark.parametrize("ga", [13.9, 40.1, 0, -5])
def test_out_of_range_gestational_age_rejected(ga):
    with pytest.raises(ValueError):
        hc_mean_mm(ga)


def test_sd_increases_with_gestational_age():
    assert hc_sd_mm(14) < hc_sd_mm(27) < hc_sd_mm(40)


def test_curves_cover_14_to_40_weeks_and_are_ordered():
    c = centile_curves(step_weeks=1.0)
    assert c[0]["ga_weeks"] == 14 and c[-1]["ga_weeks"] == 40 and len(c) == 27
    assert all(p["p3"] < p["p10"] < p["p50"] < p["p90"] < p["p97"] for p in c)
