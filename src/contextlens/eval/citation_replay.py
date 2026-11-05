"""What the citation guard would change in saved cards, with no model and no network.

The guard (`guard/citations.py`) drops a finding that cites a turn the instruction patterns flag. When the patterns grow,
findings saved as kept may now be dropped. This reruns the guard over the rows of saved E2 and E3 cards records and counts,
per system, the kept findings that would now be dropped, the cards that would change, and the cards that would end up
abstaining. Turns are named by id and pattern class only: no transcript text is printed.

Run: .venv/Scripts/python -m contextlens.eval.citation_replay <run id> [<run id> ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from contextlens.eval.records import ROOT, load_verified
from contextlens.guard.citations import check_findings
from contextlens.guard.instructions import instruction_kind
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario

RUNS_REL = Path("results/runs")


def replay_system(scenarios: list[Scenario], rows: list[Prediction]) -> dict[str, Any]:
    by_id = {s.id: s for s in scenarios}
    kept_total = dropped = other = changed = abstain = 0
    causes: Counter[tuple[str, str]] = Counter()
    for p in rows:
        if p.card is None or not p.card.findings:
            continue
        s = by_id[p.scenario_id]
        store = InMemoryTranscriptStore()
        for t in s.conversation.turns:
            store.add(t)
        kept, drops = check_findings(p.card, store, s.request.as_of_seq, s.conversation.id)
        turns = {t.id: t for t in s.conversation.turns}
        kept_total += len(p.card.findings)
        new = [f for f, why in drops if why == "instruction"]
        other += sum(why != "instruction" for _, why in drops)
        dropped += len(new)
        for f in new:
            for e in f.evidence:
                t = turns.get(e.turn_id)
                if t is not None and (kind := instruction_kind(t.text)):
                    causes[(t.id, kind)] += 1
        if new:
            changed += 1
            abstain += not kept and p.card.status != "abstained"
    return {"cards_with_findings": sum(p.card is not None and bool(p.card.findings) for p in rows),
            "findings_kept_in_the_record": kept_total, "findings_now_dropped_for_an_instruction": dropped,
            "findings_dropped_for_another_reason": other, "cards_that_would_change": changed,
            "cards_that_would_end_up_abstaining": abstain,
            "turns_causing_it": [{"turn_id": t, "pattern_class": k, "findings": n} for (t, k), n in sorted(causes.items())]}


def replay_record(record: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    if record.get("config", {}).get("part") == "consistency":
        raise ValueError(f"{record['run_id']} is a consistency record: its cards hold labels, not cited findings")
    scenarios, _ = load_verified(record["scenario_source"], record["split"], root)
    return {name: replay_system(scenarios, [Prediction.model_validate_json(ln)
                                            for ln in (root / rel).read_text(encoding="utf-8").splitlines() if ln.strip()])
            for name, rel in record["predictions"].items()}


def main(argv: list[str] | None = None) -> int:
    ids = sys.argv[1:] if argv is None else argv
    if not ids:
        print("usage: citation_replay <run id> [<run id> ...]", file=sys.stderr)
        return 2
    try:
        out = {i: replay_record(json.loads((ROOT / RUNS_REL / f"{i}.json").read_text(encoding="utf-8"))) for i in ids}
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"citation replay stopped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
