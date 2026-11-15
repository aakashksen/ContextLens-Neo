"""Text helpers for `team_rag`: clean transcript text for the prompt, and pick a short exact span of a real turn to quote.

`clean` is for prompts only. A quote must be a substring of the real turn (the citation guard compares whitespace- and
case-normalized text), so `quote_span` cuts the real text at sentence ends and at disfluency markers and never rewrites it."""

import re
from collections.abc import Iterable

_MARKER = re.compile(r"\{[^{}]{1,40}\}")  # {disfmarker}, {vocalsound}, {gap}, ...
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
_WORD = re.compile(r"[a-z0-9']+")
STOP = frozenset(
    "a an the and or but if of to in on at for with by from as is are was were be been it its this that these those i you he she "
    "we they me him her us them my your our their what which who whom whose when where why how did do does done say said tell told "
    "about discuss discussed discussion meeting summarize summary think thought view views opinion there so not no yes can could "
    "would should will just also than then".split()
)


def clean(text: str) -> str:
    """The turn with disfluency markers taken out and spaced punctuation closed up, for the model to read."""
    out = _MARKER.sub(" ", text)
    out = re.sub(r"\s+([,.;:?!])", r"\1", out)
    out = re.sub(r"(?:\s*-\s*-)+\s*", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


def segments(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of the sentences of `text`, cut also at markers; every slice is a marker-free substring."""
    cuts, pos = [], 0
    for m in _MARKER.finditer(text):
        cuts.append((pos, m.start()))
        pos = m.end()
    cuts.append((pos, len(text)))
    out = []
    for a, b in cuts:
        piece_start = a
        for m in _SENTENCE_END.finditer(text, a, b):
            out.append((piece_start, m.start()))
            piece_start = m.end()
        out.append((piece_start, b))
    return [(a, b) for a, b in ((a, b) for a, b in out) if text[a:b].strip()]


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _joinable(text: str, prev_end: int, nxt_start: int) -> bool:
    return not text[prev_end:nxt_start].strip()  # only whitespace between: one contiguous substring


def quote_span(text: str, focus: Iterable[str], max_chars: int = 240) -> str:
    """A short exact substring of `text`: the sentence that shares most `focus` words, grown to its neighbours (when only
    whitespace separates them) while it stays within `max_chars`. If no sentence shares a word, the first one."""
    segs = [_trim(text, a, b) for a, b in segments(text)]
    if not segs:
        return text.strip()[:max_chars]
    want = set(words(" ".join(focus)))
    score = [len(want & set(words(text[a:b]))) for a, b in segs]
    best = max(range(len(segs)), key=lambda i: (score[i], -i))
    lo = hi = best
    while True:
        grew = False
        for j in (hi + 1, lo - 1):  # after first: an answer follows its cue more often than it precedes it
            if 0 <= j < len(segs) and score[j] > 0:
                a, b = (segs[lo][0], segs[j][1]) if j > hi else (segs[j][0], segs[hi][1])
                ok = _joinable(text, segs[hi][1], segs[j][0]) if j > hi else _joinable(text, segs[j][1], segs[lo][0])
                if ok and b - a <= max_chars:
                    lo, hi, grew = min(lo, j), max(hi, j), True
                    break
        if not grew:
            break
    a, b = segs[lo][0], segs[hi][1]
    if b - a > max_chars:  # one long sentence: cut at a word boundary
        cut = text.rfind(" ", a, a + max_chars)
        b = cut if cut > a else a + max_chars
    return text[a:b].strip()


def clip_for_prompt(text: str, focus: Iterable[str], max_chars: int = 600) -> str:
    """The cleaned turn, shortened to about `max_chars` around its best-matching sentences when it is longer."""
    cleaned = clean(text)
    if len(cleaned) <= max_chars:
        return cleaned
    segs = [_trim(cleaned, a, b) for a, b in segments(cleaned)]
    want = set(words(" ".join(focus)))
    score = [len(want & set(words(cleaned[a:b]))) for a, b in segs]
    best = max(range(len(segs)), key=lambda i: (score[i], -i))
    lo = hi = best
    size = segs[best][1] - segs[best][0]
    while size < max_chars:
        options = [j for j in (hi + 1, lo - 1) if 0 <= j < len(segs)]
        if not options:
            break
        j = max(options, key=lambda j: (score[j], j > hi))
        add = segs[j][1] - segs[j][0] + 1
        if size + add > max_chars:
            break
        lo, hi, size = min(lo, j), max(hi, j), size + add
    out = cleaned[segs[lo][0] : segs[hi][1]]
    return ("... " if lo > 0 else "") + out + (" ..." if hi < len(segs) - 1 else "")
