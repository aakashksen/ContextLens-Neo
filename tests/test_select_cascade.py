import json
from pathlib import Path

import pytest

from contextlens.consistency import select_cascade as sc
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn

LABELS = ("consistent", "update", "contradiction", "insufficient")


def scn(sid: str, gold: str | None) -> Scenario:
    turns = [Turn(id=f"{sid}:{i}", conversation_id=f"c-{sid}", seq=i, speaker="AB"[i % 2], text=f"text {sid} {i}",
                  start_ms=i, end_ms=i + 1, source="dataset") for i in range(3)]
    return Scenario(id=sid, conversation=Conversation(id=f"c-{sid}", turns=turns, source="decode", split="dev"),
                    request=ButtonRequest(conversation_id=f"c-{sid}", as_of_seq=2), category="decode",
                    gold_evidence_turn_ids=[], gold_label=gold, gold_answer=None, origin="dataset")  # type: ignore[arg-type]


def row(sid: str, p_contra: float) -> dict:
    rest = (1 - p_contra) / 3
    return {"id": sid, "probs": {"consistent": rest, "update": rest, "contradiction": p_contra, "insufficient": rest}}


def pred(sid: str, label: str) -> Prediction:
    return Prediction(scenario_id=sid, label=label)  # type: ignore[arg-type]


def cases(spec: list[tuple[str, bool, float, bool]]) -> list[sc.Case]:
    """(source, is contradiction, trained probability, prompted says contradiction)"""
    return [sc.Case(src, f"{src}-{i}", gold, p, prompted) for i, (src, gold, p, prompted) in enumerate(spec)]


# two sources, 4 real contradictions and 4 others each
SPEC = [(s, True, p, pr) for s in ("decode", "scenarios") for p, pr in [(0.9, True), (0.8, True), (0.6, True), (0.3, False)]] + \
       [(s, False, p, pr) for s in ("decode", "scenarios") for p, pr in [(0.7, False), (0.4, True), (0.2, False), (0.05, False)]]


def test_the_catch_floor_is_a_named_constant_with_its_reason_in_the_module() -> None:
    assert sc.CATCH_FLOOR == 0.5
    assert "floor" in (sc.__doc__ or "").lower()


def test_alert_needs_the_flag_and_for_the_cascade_the_prompted_checker() -> None:
    c = sc.Case("decode", "x", False, 0.4, True)
    assert sc.alert(c, "trained", 0.4) and sc.alert(c, "cascade", 0.4)
    assert not sc.alert(c, "trained", 0.41) and not sc.alert(c, "cascade", 0.41)
    assert not sc.alert(sc.Case("decode", "y", False, 0.9, False), "cascade", 0.5)


def test_sweep_counts_catch_and_false_alert_per_source() -> None:
    pts = {(p.system, p.threshold): p for p in sc.sweep(cases(SPEC), thresholds=(0.0, 0.5))}
    t = pts[("trained", 0.5)]
    assert t.catch["decode"] == (3, 4) and t.far["decode"] == (1, 4)   # 0.9 0.8 0.6 caught; 0.7 false
    c = pts[("cascade", 0.5)]
    assert c.catch["decode"] == (3, 4) and c.far["decode"] == (0, 4)   # 0.7 was not confirmed by the prompted checker
    assert pts[("cascade", 0.0)].catch["decode"] == (3, 4)             # t=0 is the prompted checker alone
    assert pts[("cascade", 0.0)].far["decode"] == (1, 4)


def test_selection_is_lowest_false_alert_rate_among_points_meeting_the_floor_on_every_source() -> None:
    pts = sc.sweep(cases(SPEC), thresholds=(0.0, 0.25, 0.5, 0.75, 0.95))
    best = sc.select(pts, floor=0.5)
    assert best is not None and (best.system, best.threshold) == ("cascade", 0.5)  # catch 3/4, no false alert
    # a floor of 1.0 is met only by points flagging everything the models can see: prompted-alone misses a 0.3 case
    assert sc.select(pts, floor=1.0) is None


def test_a_point_that_misses_the_floor_on_one_source_is_not_eligible() -> None:
    spec = [("decode", True, 0.9, True)] * 4 + [("decode", False, 0.1, False)] * 4 + \
           [("scenarios", True, 0.1, False)] * 4 + [("scenarios", False, 0.1, False)] * 4
    assert sc.select(sc.sweep(cases(spec), thresholds=(0.5,)), floor=0.5) is None


def test_ties_go_to_higher_catch_then_the_cheaper_trained_only_system() -> None:
    spec = [(s, True, 0.9, True) for s in ("decode", "scenarios")] * 2 + [(s, False, 0.1, False) for s in ("decode", "scenarios")] * 2
    best = sc.select(sc.sweep(cases(spec), thresholds=(0.5,)), floor=0.5)
    assert best is not None and best.system == "trained"


def test_alert_precision_is_calculated_from_catch_rate_false_alert_rate_and_base_rate() -> None:
    # 80% caught, 10% false alerts, 5% of statements real contradictions: 0.04 / (0.04 + 0.095)
    assert sc.alert_precision(0.8, 0.1, 0.05) == pytest.approx(0.04 / 0.135)
    assert sc.alert_precision(0.8, 0.1, 0.20) == pytest.approx(0.16 / 0.24)
    assert sc.alert_precision(0.0, 0.0, 0.05) is None


def test_cases_align_by_id_and_a_mismatch_stops_the_run() -> None:
    dev = {"decode": [scn("a", "contradiction"), scn("b", "consistent"), scn("c", None)]}
    rescored = {"decode": [row("a", 0.9), row("b", 0.1), row("c", 0.5)]}
    prompted = {"decode": [pred("a", "contradiction"), pred("b", "consistent"), pred("c", "consistent")]}
    got = sc.build_cases(dev, rescored, prompted)
    assert [(c.id, c.contradiction, c.prompted_alert) for c in got] == [("a", True, True), ("b", False, False)]  # unlabelled c is not scored
    with pytest.raises(ValueError, match="ids"):
        sc.build_cases(dev, rescored, {"decode": [pred("a", "contradiction"), pred("b", "consistent")]})


def test_false_alarm_sample_is_fixed_by_the_seed_and_drawn_from_the_selected_point() -> None:
    spec = [("decode", False, 0.9, True)] * 40 + [("decode", True, 0.9, True)] * 5 + [("decode", False, 0.1, True)] * 5
    cs = cases(spec)
    point = sc.sweep(cs, thresholds=(0.5,))[1]
    a = sc.sample_false_alarms(cs, point, n=30, seed=13)
    assert a == sc.sample_false_alarms(cs, point, n=30, seed=13) and len({c.id for c in a}) == 30
    assert all(not c.contradiction and sc.alert(c, point.system, point.threshold) for c in a)
    assert {c.id for c in a} != {c.id for c in sc.sample_false_alarms(cs, point, n=30, seed=14)}
    assert len(sc.sample_false_alarms(cs[:3], point, n=30, seed=13)) == 3


def test_config_names_the_floor_and_the_point_and_holds_no_measured_numbers(tmp_path: Path) -> None:
    pts = sc.sweep(cases(SPEC), thresholds=(0.5,))
    point = sc.select(pts)
    out = tmp_path / "cascade.json"
    sc.write_config(out, point, {"checkpoint_sha256": "ab"})
    cfg = json.loads(out.read_text())
    assert cfg["system"] == point.system and cfg["flag_threshold"] == 0.5 and cfg["catch_floor"] == sc.CATCH_FLOOR
    assert cfg["selection_data"].startswith("dev") and cfg["provisional"] is True
    assert not {"catch", "far", "false_alert_rate", "catch_rate"} & set(cfg)


def test_report_labels_precision_as_calculated_and_shows_counts() -> None:
    pts = sc.sweep(cases(SPEC), thresholds=(0.5,))
    text = sc.report(sc.select(pts), pts)
    assert "calculated, not measured" in text and "5%" in text and "20%" in text
    assert "3/4" in text  # counts behind each rate
