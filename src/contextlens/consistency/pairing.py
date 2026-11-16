"""Pairing for contradiction/update detection: choose the few earlier turns most likely to be the statement the latest
turn conflicts with, check the latest turn against each one alone, and report the strongest conflict.

Why: the trained checker reads one statement against a pile of up to 12 earlier turns, so the one conflicting turn is
diluted. Here candidates are ranked by the latest turn's key content (numbers, names, nouns; fillers removed), same-speaker
and search-rank boosts, and every top candidate gets its own pairwise call (one batched forward pass per model).

`find_conflict(...)` is the entry point. It returns what `precheck.prepare` stores: `(label, confidence, earlier_ids)`.
"""
import math
import threading
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from contextlens.consistency.model import Example, example_from_turns, predict_proba
from contextlens.schemas import ConsistencyLabel, Turn

Label = ConsistencyLabel
STOP = frozenset("""a an the and or but so if then than that this these those it its is are was were be been being am do does
did done have has had having i me my we our us you your he she they them their his her of to in on at by for with from as
into about over after before up down out off not no nor too very just also only own same such can could will would should
shall may might must there here what which who whom when where why how all any both each few more most other some yes yeah
yep right correct okay ok well um uh mm hmm oh actually basically really maybe think mean know guess sure like kind sort
thing things got get gets going gonna wanna let lets say said says one""".split())
NUMBER_WORDS = frozenset("zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty "
                         "hundred thousand million monday tuesday wednesday thursday friday saturday sunday".split())

SAME_SPEAKER = 0.15  # added to the 0..1 overlap score
SEARCH_BONUS = 0.10  # added when the turn is among the retriever's hits for the key-content query
MIN_OVERLAP = 0.05  # a candidate must share some of the latest turn key content (stray one-word overlaps score low)
RECENT = 0.02  # tie-break: newer turns first, scaled by position in the pool


def key_terms(text: str) -> list[str]:
    """Content tokens in order of appearance: stop words and discourse fillers dropped, light plural stripping."""
    out = []
    for w in re.findall(r"[a-z0-9]+(?:[.,][a-z0-9]+)*", text.lower().replace("'s ", " ")):
        if w in STOP or (len(w) < 2 and not w.isdigit()):
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def key_query(text: str) -> str:
    return " ".join(dict.fromkeys(key_terms(text)))


def _weight(term: str, idf: float) -> float:
    return idf * (1.6 if any(c.isdigit() for c in term) or term in NUMBER_WORDS else 1.0)


def scored_candidates(statement: Turn, pool: list[Turn], search_ids: set[str] | None = None) -> list[tuple[float, float, Turn]]:
    """(score, overlap, turn) for every pool turn, best score first. `overlap` is the share of the statement's key-term
    weight (IDF-weighted; numbers and days count more) that the turn also contains, 0..1. Score adds a same-speaker boost
    (only when there is some overlap), a boost for retriever hits, and a small preference for newer turns."""
    pool = [t for t in pool if t.seq < statement.seq and t.id != statement.id]
    if not pool:
        return []
    q = set(key_terms(statement.text))
    docs = {t.id: set(key_terms(t.text)) for t in pool}
    df = Counter(w for d in docs.values() for w in d)
    n = len(pool)
    idf = {w: math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5)) for w in q}
    total = sum(_weight(w, idf[w]) for w in q) or 1.0
    ranked = []
    for i, t in enumerate(pool):
        overlap = sum(_weight(w, idf[w]) for w in q & docs[t.id]) / total
        s = overlap + (SAME_SPEAKER if t.speaker == statement.speaker and overlap else 0.0)
        s += SEARCH_BONUS if search_ids and t.id in search_ids and overlap else 0.0
        ranked.append((s + RECENT * i / n, overlap, t))
    ranked.sort(key=lambda x: -x[0])
    return ranked


def rank_candidates(statement: Turn, pool: list[Turn], search_ids: set[str] | None = None, k: int = 4,
                    min_overlap: float = MIN_OVERLAP) -> list[Turn]:
    """The up to `k` pool turns most likely to be what `statement` restates, contradicts or updates, best first; turns
    sharing less than `min_overlap` of the statement's key content are never candidates."""
    return [t for _, o, t in scored_candidates(statement, pool, search_ids) if o >= min_overlap][:k]


# ------------------------------------------------------------------ pairwise scoring

@dataclass
class PairScore:
    earlier: Turn
    trained: dict[str, float] = field(default_factory=dict)  # the trained checker, statement against this turn alone
    nli: list[dict[str, float]] = field(default_factory=list)  # one entailment/neutral/contradiction dict per NLI model


class NLIModel:
    """A public NLI cross-encoder as a second opinion. Premise: the earlier turn; hypothesis: the latest turn."""

    def __init__(self, name: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.labels = [str(self.model.config.id2label[i]).lower() for i in range(self.model.config.num_labels)]

    def probabilities(self, statement: Turn, earlier: list[Turn]) -> list[dict[str, float]]:
        import torch

        if not earlier:
            return []
        with torch.no_grad():
            enc = self.tok([t.text for t in earlier], [statement.text] * len(earlier), truncation=True, max_length=192,
                           padding=True, return_tensors="pt").to(self.device)
            p = torch.softmax(self.model(**enc).logits.float(), dim=-1).tolist()
        return [dict(zip(self.labels, row, strict=True)) for row in p]


def score_pairs(trained: Any, nlis: list[NLIModel], statement: Turn, cands: list[Turn]) -> list[PairScore]:
    """One batched forward pass per model over (statement, one earlier turn) pairs."""
    if not cands:
        return []
    exs: list[Example] = [example_from_turns(statement, [c]) for c in cands]
    rows = predict_proba(trained.model, trained.tokenizer, exs, batch_size=len(exs))
    per_nli = [m.probabilities(statement, cands) for m in nlis]
    return [PairScore(c, dict(zip(trained.labels, rows[i], strict=True)), [pn[i] for pn in per_nli])
            for i, c in enumerate(cands)]


# ------------------------------------------------------------------ gates and decision

FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'll|i'd|me|my|mine|myself)\b", re.I)
HEDGE = re.compile(r"\b(joking|kidding|jokes?|hypothetical(?:ly)?|suppose|supposing|imagine|what if|if i were|if it were)\b",
                   re.I)


def is_hedged(text: str) -> bool:
    """Jokes and hypotheticals do not assert anything, so they cannot contradict an earlier statement."""
    return bool(HEDGE.search(text))


def same_claim_possible(statement: Turn, earlier: Turn) -> bool:
    """Two first-person claims by different speakers are about different people ("I'm coming" / "I'm not coming"),
    so they cannot conflict."""
    if statement.speaker == earlier.speaker:
        return True
    return not (FIRST_PERSON.search(statement.text) and FIRST_PERSON.search(earlier.text))


NEGATION = re.compile(r"\b(no|not|never|none|nobody|nothing|nowhere|neither|nor|cannot)\b|n't", re.I)


def restates(statement: Turn, earlier: Turn) -> bool:
    """The statement repeats all of the earlier turn's key content and does not negate it: a confirmation, not a change."""
    if NEGATION.search(statement.text) and not NEGATION.search(earlier.text):
        return False
    e = set(key_terms(earlier.text))
    return bool(e) and e <= set(key_terms(statement.text))


@dataclass(frozen=True)
class Rule:
    """How pair scores become a verdict.
    With NLI models: a pair conflicts when their mean P(contradiction) reaches `nli_threshold`; it is then an update when
    the trained model's P(update) on the pair is at least `update_split`, otherwise a contradiction. A pair that NLI does
    not flag is still reported as an update when the trained model's P(update) reaches `update_rescue` (NLI often reads
    "moved to Thursday" after "Monday" as neutral, so the trained model's update signal is needed).
    Without NLI: the pair conflicts when the trained model's P(contradiction)+P(update) reaches `trained_threshold`.
    The strongest conflicting pair decides; with none, the answer is consistent (insufficient when the trained model
    prefers that on the best pair, or when there is no candidate)."""
    nli_threshold: float = 0.5
    update_split: float = 0.5
    update_rescue: float | None = 0.9
    trained_threshold: float = 0.5
    min_overlap: float = MIN_OVERLAP
    speaker_gate: bool = True
    hedge_gate: bool = True
    restate_gate: bool = True


def pair_verdict(p: PairScore, rule: Rule, statement: Turn | None = None) -> tuple[Label | None, float, int]:
    """(label or None, confidence, tier). Tier 1 is a conflict the NLI models (or, without NLI, the trained model)
    support; tier 0 is a rescued update, which only counts when no tier 1 pair exists."""
    t = p.trained
    if p.nli:
        nc = sum(d.get("contradiction", 0.0) for d in p.nli) / len(p.nli)
        if nc >= rule.nli_threshold:
            return ("update" if t.get("update", 0.0) >= rule.update_split else "contradiction"), nc, 1
        if (rule.update_rescue is not None and t.get("update", 0.0) >= rule.update_rescue
                and not (rule.restate_gate and statement is not None and restates(statement, p.earlier))):
            return "update", t["update"], 0
        return None, nc, 0
    c = t.get("contradiction", 0.0) + t.get("update", 0.0)
    if c >= rule.trained_threshold:
        return ("update" if t.get("update", 0.0) > t.get("contradiction", 0.0) else "contradiction"), c, 1
    return None, c, 0


def decide(pairs: list[PairScore], rule: Rule, statement: Turn | None = None) -> tuple[Label, float, list[str]]:
    verdicts = [(pair_verdict(p, rule, statement), p) for p in pairs]
    flagged = [(tier, c, lab, p) for (lab, c, tier), p in verdicts if lab is not None]
    if flagged:
        _, c, lab, p = max(flagged, key=lambda x: x[:2])
        return lab, c, [p.earlier.id]
    if not pairs:
        return "insufficient", 0.0, []
    best = pairs[0].trained
    if best.get("insufficient", 0.0) > best.get("consistent", 0.0):
        return "insufficient", best["insufficient"], []
    return "consistent", 1.0 - max(c for (_, c, _), _ in verdicts), []


# ------------------------------------------------------------------ entry point

_MODELS: dict[Any, Any] = {}
_SCORING = threading.Lock()  # one find_conflict at a time: the models and their tokenizers are shared
_LOADING = threading.Lock()


def load_models(checker_dir: str, nli_names: tuple[str, ...] = (), device: str | None = None) -> tuple[Any, list[NLIModel]]:
    """The trained checker and the named NLI models, loaded once per process (on `device`, default: the GPU if it fits)."""
    with _LOADING:
        return _load_models(checker_dir, nli_names, device)


def _load_models(checker_dir: str, nli_names: tuple[str, ...], device: str | None) -> tuple[Any, list[NLIModel]]:
    from pathlib import Path

    from contextlens.consistency.model import TrainedChecker

    if "trained" not in _MODELS:
        _MODELS["trained"] = TrainedChecker.from_checkpoint(Path(checker_dir), device=device, strip_instructions=False)
    for n in nli_names:
        if ("nli", n) not in _MODELS:
            _MODELS["nli", n] = NLIModel(n, device)
    return _MODELS["trained"], [_MODELS["nli", n] for n in nli_names]


def find_conflict(tools: Any, statement_turn: Turn, earlier_turns: list[Turn], *, trained: Any,
                  nlis: list[NLIModel] | None = None, rule: Rule = Rule(), top_n: int = 3,
                  search_k: int = 8) -> tuple[Label, float, list[str]]:
    """(label, confidence, earlier_turn_ids) for `statement_turn` against `earlier_turns`.

    `tools` is anything with `search_transcript(query, k)` returning scored turns (agents.tools.Tools), or None: turns
    the search finds for the statement's key content are added to the pool and boosted. `earlier_turn_ids` holds the one
    turn the verdict rests on when a conflict is reported (the pair to cite), otherwise it is empty. A failed search is
    skipped; model failures propagate (the caller, like precheck, records them and flags nothing)."""
    if not _SCORING.acquire(blocking=False):
        raise RuntimeError("the conflict finder is busy")  # precheck records it and flags nothing
    try:
        return _find_conflict(tools, statement_turn, earlier_turns, trained=trained, nlis=nlis, rule=rule, top_n=top_n,
                              search_k=search_k)
    finally:
        _SCORING.release()


def _find_conflict(tools: Any, statement_turn: Turn, earlier_turns: list[Turn], *, trained: Any,
                  nlis: list[NLIModel] | None = None, rule: Rule = Rule(), top_n: int = 3,
                  search_k: int = 8) -> tuple[Label, float, list[str]]:
    """(label, confidence, earlier_turn_ids) for `statement_turn` against `earlier_turns`.

    `tools` is anything with `search_transcript(query, k)` returning scored turns (agents.tools.Tools), or None: turns
    the search finds for the statement's key content are added to the pool and boosted. `earlier_turn_ids` holds the one
    turn the verdict rests on when a conflict is reported (the pair to cite), otherwise it is empty. A failed search is
    skipped; model failures propagate (the caller, like precheck, records them and flags nothing)."""
    if rule.hedge_gate and is_hedged(statement_turn.text):
        return "consistent", 1.0, []
    pool = {t.id: t for t in earlier_turns}
    hit_ids: set[str] = set()
    if tools is not None:
        try:
            for h in tools.search_transcript(key_query(statement_turn.text) or statement_turn.text, search_k):
                if h.turn.seq < statement_turn.seq:
                    pool.setdefault(h.turn.id, h.turn)
                    hit_ids.add(h.turn.id)
        except Exception:
            pass
    cands = rank_candidates(statement_turn, list(pool.values()), hit_ids, top_n * 2, rule.min_overlap)
    if rule.speaker_gate:
        cands = [c for c in cands if same_claim_possible(statement_turn, c)]
    return decide(score_pairs(trained, nlis or [], statement_turn, cands[:top_n]), rule, statement_turn)
