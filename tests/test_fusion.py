"""Tests for the rule-based screening summary (ml/fusion.py): every rule R1-R6 and the edge cases."""
import pytest

from ml.fusion import CAVEATS, ScreeningInput, summarize

ALL_NORMAL = dict(plane_status="pass", hc_z_score=0.3, hc_centile=61.8, low_reliability=False, ctg_class="Normal",
                  ctg_probability=0.98)


def test_routine_when_everything_unremarkable():
    s = summarize(ScreeningInput(**ALL_NORMAL))
    assert s.level == "Routine" and s.drivers == [] and s.not_assessed == []
    assert len(s.unremarkable) == 4


def test_R1_pathological_ctg_always_escalates():
    s = summarize(ScreeningInput(**{**ALL_NORMAL, "ctg_class": "Pathological", "ctg_probability": 0.97}))
    assert s.level == "Urgent review"
    assert s.drivers[0].rule == "R1" and "Pathological (p = 0.97)" in s.drivers[0].finding


def test_R1_wins_over_review_findings():
    s = summarize(ScreeningInput(plane_status="warn", hc_z_score=-2.5, low_reliability=True,
                                 ctg_class="Pathological"))
    assert s.level == "Urgent review"
    assert [d.rule for d in s.drivers][0] == "R1" and {d.rule for d in s.drivers} == {"R1", "R3", "R4", "R5"}


def test_R2_suspect_ctg():
    assert summarize(ScreeningInput(**{**ALL_NORMAL, "ctg_class": "Suspect"})).level == "Review recommended"


@pytest.mark.parametrize("z, side", [(-1.9, "below the 3rd"), (2.4, "above the 97th")])
def test_R3_hc_outside_p3_p97(z, side):
    s = summarize(ScreeningInput(**{**ALL_NORMAL, "hc_z_score": z}))
    assert s.level == "Review recommended" and s.drivers[0].rule == "R3" and side in s.drivers[0].finding


def test_R3_boundary_is_inclusive_of_normal_range():
    assert summarize(ScreeningInput(**{**ALL_NORMAL, "hc_z_score": 1.88})).level == "Routine"
    assert summarize(ScreeningInput(**{**ALL_NORMAL, "hc_z_score": -1.88})).level == "Routine"


@pytest.mark.parametrize("extra, text", [({"measurement_withheld": True, "hc_z_score": None}, "withheld"),
                                         ({"plane_overridden": True}, "override")])
def test_R4_plane_warning(extra, text):
    s = summarize(ScreeningInput(**{**ALL_NORMAL, "plane_status": "warn", **extra}))
    assert s.level == "Review recommended" and s.drivers[0].rule == "R4" and text in s.drivers[0].finding


def test_R5_low_reliability():
    s = summarize(ScreeningInput(**{**ALL_NORMAL, "low_reliability": True}))
    assert s.level == "Review recommended" and s.drivers[0].rule == "R5"


def test_missing_inputs_are_listed_not_assumed_normal():
    s = summarize(ScreeningInput(ctg_class="Normal"))
    assert s.level == "Routine"
    assert len(s.not_assessed) == 3 and "Standard-plane check" in s.not_assessed


def test_ultrasound_only_and_ctg_only():
    us = summarize(ScreeningInput(plane_status="pass", low_reliability=False))
    assert us.level == "Routine" and any("CTG" in m for m in us.not_assessed)
    assert summarize(ScreeningInput(ctg_class="Pathological")).level == "Urgent review"


def test_no_input_is_insufficient_data():
    s = summarize(ScreeningInput())
    assert s.level == "Insufficient data" and len(s.not_assessed) == 4


def test_unavailable_plane_check_is_not_assessed():
    s = summarize(ScreeningInput(plane_status="unavailable", ctg_class="Normal"))
    assert "Standard-plane check" in s.not_assessed


def test_caveat_states_different_patients():
    assert any("different patients" in c for c in summarize(ScreeningInput(**ALL_NORMAL)).caveats)
    assert CAVEATS[0].startswith("The ultrasound and CTG models were trained on different patients")


def test_unknown_values_rejected():
    with pytest.raises(ValueError):
        summarize(ScreeningInput(ctg_class="Bad"))
    with pytest.raises(ValueError):
        summarize(ScreeningInput(plane_status="maybe"))
