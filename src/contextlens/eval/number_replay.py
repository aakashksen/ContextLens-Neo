"""What the corrected number rule would have kept: the saved `drops` of a record, checked again, with no model call.

A drop holds the statement and the ids of the turns it cited, not its quotes. A quote is part of its turn, so the
numbers of the cited turns are a superset of the numbers of the quotes: a drop that still fails against the whole
turns would fail against the quotes too, and one that passes may fail against the quotes. The counts of survivors are
therefore upper bounds.

Run: python -m contextlens.eval.number_replay RUN_ID [RUN_ID ...]
"""

import json
import sys
from pathlib import Path
from typing import Any

from contextlens.agents.single import numbers_supported
from contextlens.data.scenarios import load_scenarios
from contextlens.eval.records import ROOT, RUNS_REL, SCENARIO_SOURCES
from contextlens.replay.runner import Prediction


def replay_record(run_id: str, root: Path = ROOT) -> dict[str, Any]:
    """Per system of a record: its number drops, how many pass the corrected rule against the turns they cited, and
    the cases whose card ended abstained with only such drops behind it (answerable gold, not insufficient)."""
    rec = json.loads((root / RUNS_REL / f"{run_id}.json").read_text(encoding="utf-8"))
    manifest, directory = SCENARIO_SOURCES[rec["scenario_source"]]
    scenarios = {s.id: s for s in load_scenarios(rec["split"], root / directory)}
    out: dict[str, Any] = {}
    for system, rel in rec["predictions"].items():
        rows = [Prediction.model_validate_json(ln) for ln in (root / rel).read_text(encoding="utf-8").splitlines()]
        number_drops = survivors = 0
        revived: list[str] = []
        saved = any(p.drops for p in rows)
        for p in rows:
            sc = scenarios[p.scenario_id]
            texts = {t.id: t.text for t in sc.conversation.turns if t.seq <= sc.request.as_of_seq}
            kept = [d for d in p.drops if d.reason == "number"
                    and numbers_supported(d.statement, [texts.get(i, "") for i in d.turn_ids], d.turn_ids)]
            number_drops += sum(d.reason == "number" for d in p.drops)
            survivors += len(kept)
            others = [d for d in p.drops if d.reason != "number"]
            if kept and not others and p.card and p.card.status == "abstained" and sc.gold_label != "insufficient":
                revived.append(sc.id)
        out[system] = {"drops_saved_in_rows": saved, "number_drops": number_drops, "would_survive": survivors,
                       "answerable_abstained_cases_with_a_surviving_drop": sorted(revived)}
    return out


def main(argv: list[str]) -> int:
    for run_id in argv:
        print(json.dumps({run_id: replay_record(run_id)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
