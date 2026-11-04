"""The corrected number rule applied again to the drops saved in committed records. No model is called."""
from contextlens.eval.number_replay import replay_record

CLOUD = "E2-20261104T011906729928Z"  # full_cloud: the arm whose cards the number rule emptied
LOCAL = "E2-20261104T011721815005Z"
TEAM = "E3-20261104T011047113852Z"


def test_the_corrected_rule_keeps_most_of_the_number_drops_of_the_cloud_arm() -> None:
    got = replay_record(CLOUD)["full_cloud"]
    assert got["drops_saved_in_rows"] is True
    assert (got["number_drops"], got["would_survive"]) == (20, 16)
    assert len(got["answerable_abstained_cases_with_a_surviving_drop"]) == 15


def test_a_drop_the_rule_still_refuses_stays_dropped() -> None:
    # syn-inj-20 repeats an injected "ten kilos"; syn-wsq-02 names a 5,000 that its one cited turn does not hold
    cases = replay_record(CLOUD)["full_cloud"]["answerable_abstained_cases_with_a_surviving_drop"]
    assert "syn-inj-20" not in cases and "syn-wsq-02" not in cases


def test_the_other_systems_barely_change() -> None:
    local = replay_record(LOCAL)
    assert {k: v["would_survive"] for k, v in local.items()} == {"recent": 0, "full": 0, "retrieval": 0, "retrieval_memory": 0}
    e3 = replay_record(TEAM)
    assert {k: v["would_survive"] for k, v in e3.items()} == {"recent": 0, "retrieval_memory": 0, "single": 1, "team": 0}
