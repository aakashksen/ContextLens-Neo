"""Turns that give orders instead of stating facts. Such a turn cannot be the evidence for a claim of fact.

A rule in code, so no model can be talked out of it. It is deliberately narrow: a speaker telling a colleague "ignore
that email" is ordinary talk, while an order aimed at the assistant (it names the assistant or the transcript, tells it
what to say, or tells it to drop its rules) is not evidence of anything but the order. It errs both ways, and
`experiments/e3_team.py` reports how often on dev."""
import re

_SENTENCE = r"(?:^|[.!?:;,]\s+|\n)(?:please\s+|now\s+|just\s+)?"
_TARGET = r"(?:instructions?|rules?|prompt|transcript|conversation|question|guidelines?|context|messages?|above|everything)"
_PATTERNS = {
    "names the assistant": r"^\W*(?:hey |hi |dear |ok |okay |attention )?(?:assistant|ai|chatbot|bot)\b\s*[,:!]"
                           r"|\b(?:to|for) (?:the )?(?:ai|assistant|chatbot)\b\s*[:,!]"
                           r"|\bas the (?:ai|assistant)\b|\byou are (?:now )?(?:an? )?(?:ai|assistant)\b"
                           r"|\b(?:developer|debug|admin|god) mode\b",
    "drops the rules": rf"\b(?:ignore|disregard|forget|override|bypass)\b[^.!?\n]{{0,30}}\b{_TARGET}\b"
                       r"|\bstop following\b|\bignore (?:all )?(?:previous|prior|earlier)\b",
    "says what to say": r"\b(?:tell|inform)\s+(?:the\s+)?(?:user|assistant)\b(?!\s+(?:group|guide|manual|interface|account|story|name))"
                        rf"|{_SENTENCE}(?:claim|reply|respond|answer|print)\s+(?:only\s+)?(?:that|with|'|\")",
    "asks for secrets": r"\breveal\b[^.!?\n]{0,40}\b(?:prompt|keys?|codes?|notes?|instructions?|secrets?|passwords?)\b"
                        r"|\byour (?:hidden |system |secret )?(?:prompt|instructions|rules)\b",
    "markup or commands": r"<\s*/?\s*(?:instructions?|system|assistant)\b|\brm\s+-rf\b|\bdo not cite\b",
}
_COMPILED = {name: re.compile(rx, re.I) for name, rx in _PATTERNS.items()}


def instruction_kind(text: str) -> str | None:
    """Why `text` reads as an order to the assistant, or None when it does not."""
    return next((name for name, rx in _COMPILED.items() if rx.search(text)), None)


def is_instruction(text: str) -> bool:
    return instruction_kind(text) is not None
