"""E10: the case builder, the code-scored metrics, the claim rule, the arm check and the grading files. Fakes only: no model,
no Ollama, no dataset, no API."""
import json
from pathlib import Path

import pytest

from contextlens.data import e10_cases as C
from contextlens.eval import e10_metrics as M
from contextlens.experiments import e10_grade as G
from contextlens.experiments import e10_long as L
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Card, Conversation, EvidenceRef, Finding, Scenario, Turn

ROOT = Path(__file__).resolve().parent.parent


def _meetings(n: int = 80, length: int = 300) -> dict[str, list[Turn]]:
    out = {}
    for m in range(n):
        cid = f"ami:M{m:03d}"
        out[cid] = [Turn(id=f"{cid}:t{i}", conversation_id=cid, seq=i, speaker="ABCD"[(i * 7 + m) % 4],
                         text=f"real words number {i} of meeting {m}", start_ms=i * 1000, end_ms=i * 1000 + 900, source="dataset")
                    for i in range(length)]
    return out


def test_conflict_triplets_have_gap_speakers_and_gold():
    scs, meta = C.build_conflicts(_meetings())
    assert len(scs) == 90 and {m["triplet"] for m in meta.values()} == set(range(30))
    for sc in scs:
        m, t = meta[sc.id], sc.conversation.turns
        a, b = sc.gold_evidence_turn_ids
        assert a == f"{sc.id}:t{m['gold_seq'][0]}" and b == f"{sc.id}:t{len(t) - 1}"
        assert 20 <= m["distractor_turns"] <= 60 and m["speakers_in_window"] >= 3
        assert t[m["gold_seq"][0]].source == t[-1].source == "synthetic"
        assert sc.request.as_of_seq == len(t) - 1 and sc.request.question is None
        assert [x.id for x in t] == [f"{sc.id}:t{i}" for i in range(len(t))]
    trip = [s for s in scs if meta[s.id]["triplet"] == 0]
    assert sorted(s.gold_label for s in trip) == ["consistent", "contradiction", "update"]
    assert len({tuple(x.text for x in s.conversation.turns[:-1]) for s in trip}) == 1  # same meeting stretch and statement A
    assert sum(meta[s.id]["different_speakers"] for s in scs if s.gold_label == "update") > 0


def test_twins_differ_only_in_the_answer_turn():
    scs, meta = C.build_twins(_meetings())
    assert len(scs) == 100
    by = {s.id: s for s in scs}
    for p in range(50):
        a, i = by[f"e10-tw-{p:02d}-answerable"], by[f"e10-tw-{p:02d}-insufficient"]
        at = meta[a.id]["answer_turn"]
        assert a.request.question == i.request.question and len(a.conversation.turns) == len(i.conversation.turns) >= 40
        assert a.gold_evidence_turn_ids == [f"{a.id}:t{at}"] and i.gold_evidence_turn_ids == [] and i.gold_label == "insufficient"
        rest_a = [x.text for k, x in enumerate(a.conversation.turns) if k != at]
        rest_i = [x.text for k, x in enumerate(i.conversation.turns) if k != at]
        assert rest_a == rest_i and a.conversation.turns[at].text != i.conversation.turns[at].text and a.gold_answer
        assert a.request.as_of_seq == len(a.conversation.turns) - 1


def test_no_meeting_on_both_sides():
    _, m1 = C.build_conflicts(_meetings())
    _, m2 = C.build_twins(_meetings())
    sides: dict[str, set[str]] = {}
    for m in (*m1.values(), *m2.values()):
        sides.setdefault(m["meeting"], set()).add(m["split"])
    assert sides and all(len(v) == 1 for v in sides.values())
    assert all(C.side(k) == next(iter(v)) for k, v in sides.items())


def test_build_is_deterministic():
    a, _ = C.build_conflicts(_meetings())
    b, _ = C.build_conflicts(_meetings())
    assert [s.model_dump_json() for s in a] == [s.model_dump_json() for s in b]


def test_frozen_manifest_matches_counts():
    m = json.loads((ROOT / C.MANIFEST_REL).read_text(encoding="utf-8"))
    assert m["counts"]["dev"]["total"] + m["counts"]["test"]["total"] == 330
    assert sum(1 for c in m["cases"].values() if c["type"] == "qmsum") == 100
    assert all(set(m["splits"][s]["ids"]) == {i for i, c in m["cases"].items() if c["split"] == s} for s in ("dev", "test"))


# ---------------------------------------------------------------- scorer

def _sc(cat, gold, label=None, q=None, n=12):
    sid = "c1"
    turns = [Turn(id=f"{sid}:t{i}", conversation_id=sid, seq=i, speaker="A", text="x", start_ms=0, end_ms=1, source="dataset")
             for i in range(n)]
    return Scenario(id=sid, conversation=Conversation(id=sid, turns=turns, source="ami"), category=cat, origin="synthetic",
                    request=ButtonRequest(conversation_id=sid, as_of_seq=n - 1, question=q), gold_label=label,
                    gold_answer="a" if q else None, gold_evidence_turn_ids=[f"{sid}:t{g}" for g in gold])


def _card(findings, status="answered", missing=(), tokens=100, mode="recall"):
    fs = [Finding(agent="x", kind="consistency", statement="s", label=lab, confidence=0.5,
                  evidence=[EvidenceRef(turn_id=f"c1:t{t}", quote="q", speaker="A", start_ms=0) for t in ev]) for lab, ev in findings]
    return Prediction(scenario_id="c1", card=Card(mode=mode, status=status, findings=fs, missing=list(missing), tokens=tokens),
                      stage_latency_ms={"total": 3000})


def test_recall_citation_metrics():
    sc = _sc("qmsum_long", [1, 2], q="what?")
    r = M.case_scores(sc, _card([(None, [1, 5])]), "m")
    assert (r["cite_any"], r["cite_all"], r["precision"], r["cited_n"]) == (True, False, 0.5, 2)
    r = M.case_scores(sc, _card([(None, [1, 2])]), "m")
    assert r["cite_all"] is True and r["precision"] == 1.0
    r = M.case_scores(sc, _card([], status="abstained"), "m")
    assert r["cite_any"] is False and r["precision"] is None and r["answered"] is False


def test_conflict_catch_needs_label_and_both_turns():
    sc = _sc("conflict_contradiction", [2, 9], label="contradiction")
    assert M.case_scores(sc, _card([("contradiction", [2, 9])]), "m")["caught"] is True
    assert M.case_scores(sc, _card([("contradiction", [9])]), "m")["caught"] is False  # one side only
    r = M.case_scores(sc, _card([("update", [2, 9])]), "m")
    assert r["caught"] is False and r["flagged_both"] is True  # flagged, wrong kind
    assert M.case_scores(sc, _card([(None, [2, 9])]), "m")["flagged_both"] is False  # an unlabeled finding is no flag


def test_false_alert_on_consistent_twin():
    sc = _sc("conflict_consistent", [2, 9], label="consistent")
    assert M.case_scores(sc, _card([("update", [2])]), "m")["false_alert"] is True
    ok = M.case_scores(sc, _card([("consistent", [2, 9])]), "m")
    assert ok["false_alert"] is False and ok["label_ok"] is True
    assert M.case_scores(sc, _card([(None, [2, 9])]), "m")["label_ok"] is False  # no label reads as insufficient


def test_abstention_and_deadline_cut():
    ins = _sc("twin_insufficient", [], label="insufficient", q="q?")
    assert M.case_scores(ins, _card([], status="abstained", missing=["deadline reached"]), "m")["abstained_ok"] is True
    r = M.case_scores(ins, _card([("consistent", [1])], missing=["deadline reached"]), "m")
    assert r["abstained_ok"] is False and r["deadline_cut"] is True
    assert M.case_scores(_sc("twin_answerable", [3], q="q?"), _card([(None, [3])]), "m")["answered_ok"] is True


def test_error_row_is_scored_not_dropped():
    card = Card(mode="recall", status="abstained", findings=[], missing=["error: X"])
    r = M.case_scores(_sc("qmsum_long", [1], q="q"), Prediction(scenario_id="c1", card=card, errored=True), "m")
    assert r["errored"] is True and r["cite_any"] is False


def test_compare_pairs_and_mcnemar():
    def tab(vals):
        return [{"id": f"x{i}", "category": "qmsum_long", "group": f"g{i % 7}", "cite_any": v} for i, v in enumerate(vals)]
    a, b = tab([True] * 20), tab([i < 5 for i in range(20)])
    c = M.compare(a, b, "cite_any", resamples=500)
    assert c["n"] == 20 and c["a_only"] == 15 and c["b_only"] == 0 and c["diff"] == 0.75
    assert c["mcnemar_p"] < 0.001 and c["ci"][0] > 0
    assert M.compare(a, [], "cite_any") is None


def test_twin_pair_unit():
    tab = [{"id": "e10-tw-00-answerable", "category": "twin_answerable", "group": "g", "answered_ok": True, "cite_any": True},
           {"id": "e10-tw-00-insufficient", "category": "twin_insufficient", "group": "g", "abstained_ok": True}]
    assert M.units(tab, "twin_pair_ok") == [("e10-tw-00", "g", 1.0)]
    tab[0]["cite_any"] = False
    assert M.units(tab, "twin_pair_ok")[0][2] == 0.0


def _arm(catch, cites, false_alert=0, cut=False, lat=3000, tok=100):
    rows = []
    for i in range(40):
        common = {"group": f"m{i % 10}", "errored": False, "tokens": tok, "latency_ms": lat}
        rows.append({"id": f"cf{i}", "category": "conflict_contradiction", "caught": i < catch, "deadline_cut": cut, **common})
        rows.append({"id": f"qm{i}", "category": "qmsum_long", "cite_any": i < cites, "deadline_cut": False, **common})
        rows.append({"id": f"fa{i}", "category": "conflict_consistent", "false_alert": i < false_alert, "deadline_cut": False, **common})
        rows.append({"id": f"ct{i}", "category": "control_recall", "deadline_cut": False, **common})
    return rows


def test_claim_rule_needs_all_arms_and_all_conditions():
    assert M.claim_verdict({"single": _arm(5, 5)})["decided"] is False
    base = {"single": _arm(5, 20), "single_plus": _arm(10, 20), "team_shipping_changed": _arm(5, 20)}
    v = M.claim_verdict({**base, "team_routed": _arm(36, 20, tok=120, lat=3300)})
    assert v["decided"] and v["team_beats_single_plus_on"] == ["catch_rate"] and v["may_claim_team_catches_more"] is True
    assert v["may_add_single_agent_cost"] is True and v["cost"]["latency_ratio"] == pytest.approx(1.1)
    dear = M.claim_verdict({**base, "team_routed": _arm(36, 20, tok=400, lat=3300)})
    assert dear["may_claim_team_catches_more"] and dear["may_add_single_agent_cost"] is False
    assert M.claim_verdict({**base, "team_routed": _arm(36, 20, cut=True)})["may_claim_team_catches_more"] is False
    tie = M.claim_verdict({**base, "team_routed": _arm(10, 20)})
    assert tie["team_beats_single_plus_on"] == [] and tie["may_claim_team_catches_more"] is False
    worse = M.claim_verdict({**base, "team_routed": _arm(36, 5)})
    assert worse["team_clearly_worse_on"] == ["cite_any"] and worse["may_claim_team_catches_more"] is False


# ---------------------------------------------------------------- runner

def test_missing_arm_fails_clearly():
    if "team_routed" in L.production.ASSISTANTS:
        pytest.skip("the new arms are registered")
    with pytest.raises(L.ArmMissing, match="team_routed"):
        L.check_arms(["single", "team_routed"])
    L.check_arms(["single", "team_shipping_changed"])


def test_config_names_the_four_arms_and_matches_production():
    cfg = L.load_config(ROOT)
    assert cfg["arms"] == ["single", "single_plus", "team_shipping_changed", "team_routed"]
    L.check_pins(cfg, ROOT)


def test_saved_rows_resume_skip_errors_and_cut_lines(tmp_path):
    ok = _card([(None, [1])])
    bad = Prediction(scenario_id="c2", card=Card(mode="recall", status="abstained", findings=[], missing=["error"]), errored=True)
    p = tmp_path / L.ROWS_REL
    for arm, row, key in (("single", ok, "k"), ("single", bad, "k"), ("single", ok, "other")):
        L._append(p, json.dumps({"key": key, "split": "dev", "arm": arm, "row": json.loads(row.model_dump_json())}))
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"key": "k", "sp')  # a crash cut the last line
    got = L.saved_rows(tmp_path, "k", "dev")
    assert list(got) == ["single"] and list(got["single"]) == ["c1"]


def test_arm_is_built_by_the_product_factory_and_answers():
    """run_case on a fake writer: the factory builds the arm, memory is loaded, a card comes back, no network."""
    from contextlens.llm.client import FakeLLM

    class Emb:
        def embed(self, texts):
            return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    sc = _sc("qmsum_long", [1], q="what?")
    # a writer that fails fast: the card is an error or abstained card, never a crash of the runner
    row = L.run_case(sc, "single", FakeLLM([]), Emb(), [], 2.0)
    assert isinstance(row, Prediction) and row.card is not None


# ---------------------------------------------------------------- grading files

def _scenarios_for_grading():
    out, meta = [], {}
    for i in range(12):
        sid = f"q{i}"
        out.append(_sc("qmsum_long", [1], q="q?").model_copy(update={"id": sid}))
        meta[sid] = {"type": "qmsum", "meeting": f"m{i}"}
    for i in range(8):
        for lab in ("contradiction", "update", "consistent"):
            sid = f"cf{i}-{lab}"
            out.append(_sc(f"conflict_{lab}", [2, 9], label=lab).model_copy(update={"id": sid}))
            meta[sid] = {"type": "conflict", "meeting": f"cm{i}", "triplet": i}
    return out, meta


def test_sample_keeps_triplets_whole_and_is_seeded():
    scs, meta = _scenarios_for_grading()
    a = G.sample_cases(scs, meta, 20, 13)
    assert [s.id for s in a] == [s.id for s in G.sample_cases(scs, meta, 20, 13)]
    trip: dict[int, list[str]] = {}
    for s in a:
        if meta[s.id]["type"] == "conflict":
            trip.setdefault(meta[s.id]["triplet"], []).append(s.id)
    assert trip and all(len(v) == 3 for v in trip.values())


def test_item_hides_arm_and_shows_cited_turns():
    sc = _sc("qmsum_long", [1], q="what?")
    item = G.item_for("g1", sc, _card([(None, [1, 4])]), "qmsum_long")
    assert set(item) == {"item_id", "case_type", "question", "gold_answer", "gold_label", "gold_turns", "card", "cited_turns"}
    assert [t["turn_id"] for t in item["gold_turns"]] == ["c1:t1"] and [t["turn_id"] for t in item["cited_turns"]] == ["c1:t4"]
    assert "tokens" not in json.dumps(item) and "latency" not in json.dumps(item)


def test_graded_files_are_checked_and_summarised(tmp_path):
    key = {f"g{i}": {"arm": "single" if i % 2 == 0 else "team_routed", "case": f"c{i // 2}", "type": "qmsum", "group": f"m{i // 2}"}
           for i in range(40)}
    (tmp_path / "batch-001.json").write_text(json.dumps({"batch": 1, "items": [{"item_id": i} for i in key]}), encoding="utf-8")
    good = [{"item_id": i, "verdict": "correct" if int(i[1:]) % 4 else "wrong", "reason": "because"} for i in key]
    path = tmp_path / "graded-batch-001.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    s = G.summarize_grades(key, G.read_graded(tmp_path, key))
    assert s["n_graded"] == 40 and set(s["arms"]) == {"single", "team_routed"} and s["comparisons"][0]["n"] == 20
    for bad, msg in ((good[:-1], "no grade"), ([*good, good[0]], "twice"), ([{**good[0], "verdict": "ok"}, *good[1:]], "verdict"),
                     ([{**good[0], "reason": " "}, *good[1:]], "no reason"), ([{**good[0], "item_id": "zz"}, *good[1:]], "unknown")):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            G.read_graded(tmp_path, key)


# ---------------------------------------------------------------- memory cache and prebuild

def test_memory_key_ignores_commit_and_case_id_but_not_memory_settings():
    a = _sc("qmsum_long", [1], q="q?")
    b = a.model_copy(update={"id": "other", "conversation": a.conversation.model_copy(update={"id": "other"})})
    k = L.memory_key(a, 40, "m", 16384)
    assert k == L.memory_key(b, 40, "m", 16384)  # another question on the same meeting shares it
    assert k != L.memory_key(a, 20, "m", 16384) and k != L.memory_key(a, 40, "m", 4096) and k != L.memory_key(a, 40, "x", 16384)
    assert k != L.memory_key(a.model_copy(update={"request": a.request.model_copy(update={"as_of_seq": 5})}), 40, "m", 16384)


class _Extractor:
    """A stand-in model: one note per call, citing the first turn of the chunk it is shown; calls in `fail` raise."""

    def __init__(self, fail=()):
        self.calls, self.fail = 0, set(fail)

    def complete(self, messages, **kw):
        import re
        from types import SimpleNamespace

        from contextlens.memory import extract as X

        self.calls += 1
        if self.calls in self.fail:
            raise RuntimeError("timed out")
        first = re.search(r"\[(c1:t\d+)\]", messages[1]["content"]).group(1)
        return SimpleNamespace(parsed=X._Out(items=[X._Raw(kind="fact", subject="s", statement=f"note {first}", source_turn_ids=[first])]))


def test_failed_chunk_is_not_cached_and_only_it_is_redone(tmp_path):
    sc = _sc("qmsum_long", [1], q="q?", n=90)  # three chunks of 40, 40, 10
    seen = []
    first = _Extractor(fail={2, 3})  # chunk 2 fails on both attempts
    items, info = L.build_memory(sc, first, 40, tmp_path, 16384, lambda n, of, s: seen.append((n, of)), attempts=2)
    assert [i.as_of_seq for i in items] == [39, 89] and info["failed_chunks"] == 1 and seen == [(1, 3), (2, 3), (3, 3)]
    key = L.memory_key(sc, 40, L.production.MODEL, 16384)
    assert not L._cached(tmp_path, key)  # an unfinished entry is not reused as memory
    again = _Extractor()
    items2, info2 = L.build_memory(sc, again, 40, tmp_path, 16384)
    assert again.calls == 1 and info2["redone"] == 1 and info2["failed_chunks"] == 0  # only the failed chunk
    assert sorted(i.as_of_seq for i in items2) == [39, 79, 89] and L._cached(tmp_path, key)
    cached, info3 = L.build_memory(sc, _Extractor(), 40, tmp_path, 16384)  # later commit or case id: found by content
    assert info3["cached"] is True and cached == items2


def test_patient_client_never_shortens_a_timeout():
    seen = {}

    class Inner:
        def complete(self, messages, **kw):
            seen.update(kw)

    L.Patient(Inner(), 900).complete([], timeout_s=60.0)
    assert seen["timeout_s"] == 900
    L.Patient(Inner(), 900).complete([], timeout_s=2000.0)
    assert seen["timeout_s"] == 2000


def test_config_sets_memory_context():
    m = L.load_config(ROOT)["memory"]
    assert m["chunk_turns"] == 40 and m["num_ctx"] >= 12288


def test_claim_verdict_survives_zero_tokens_in_the_base_arm():
    zero = _arm(5, 20, tok=0)
    v = M.claim_verdict({"single": zero, "single_plus": _arm(10, 20), "team_shipping_changed": _arm(5, 20), "team_routed": _arm(36, 20)})
    assert v["decided"] and v["cost"] == {} and v["may_add_single_agent_cost"] is False


def test_timed_out_call_at_the_deadline_counts_as_a_cut():
    p = _card([], status="abstained", missing=["llm error: request to x failed: timed out"])
    assert M.deadline_cut(p) is False  # 3 s: an outage-style timeout, not the deadline
    p.stage_latency_ms["total"] = 8030
    assert M.deadline_cut(p) is True
