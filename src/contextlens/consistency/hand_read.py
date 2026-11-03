"""Write the false alarms to read by hand: HAND_READ_N chosen with a fixed seed from the selected operating point.

The file holds dataset text, so it goes to the git-ignored results/local/ and only ids are printed.

Run: python -m contextlens.consistency.hand_read --data-root <folder with datasets/raw/>
"""
import argparse
import json
import sys
from pathlib import Path

from contextlens.consistency.rescore import OUT_REL
from contextlens.consistency.select_cascade import (
    CONFIG_REL, HAND_READ_N, SEED, Case, Point, load_inputs, sample_false_alarms,
)
from contextlens.eval.records import ROOT
from contextlens.schemas import Scenario

EARLIER_SHOWN = 2


def write_hand_read(path: Path, sample: list[Case], dev: dict[str, list[Scenario]]) -> list[str]:
    by_id = {(src, s.id): s for src, scs in dev.items() for s in scs}
    lines = [f"# {len(sample)} false alarms at the selected point (seed {SEED}); local only, holds dataset text", ""]
    for k, c in enumerate(sample, 1):
        s = by_id[(c.source, c.id)]
        turns = sorted((t for t in s.conversation.turns if t.seq <= s.request.as_of_seq), key=lambda t: t.seq)
        lines += [f"## {k}. {c.id} ({c.source})", f"- gold: {s.gold_label}; trained P(contradiction) {c.p_contradiction:.3f}",
                  *(f"- {t.speaker}: {t.text}" for t in turns[-(EARLIER_SHOWN + 1):-1]),
                  f"- statement, {turns[-1].speaker}: {turns[-1].text}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return [c.id for c in sample]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    cfg = json.loads((args.root / CONFIG_REL).read_text(encoding="utf-8"))
    dev, cases = load_inputs(args.root, args.data_root)
    point = Point(cfg["system"], cfg["flag_threshold"], {}, {})
    out = args.root / OUT_REL / "false-alarms.md"
    ids = write_hand_read(out, sample_false_alarms(cases, point, HAND_READ_N, SEED), dev)
    print(f"{len(ids)} false alarms written to {out}")
    print("\n".join(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
