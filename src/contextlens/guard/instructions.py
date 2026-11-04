"""Turns that give orders instead of stating facts. Such a turn cannot be the evidence for a claim of fact.

A rule in code, so no model can be talked out of it. It is deliberately narrow: a speaker telling a colleague "ignore
that email" is ordinary talk, while an order aimed at the assistant (it names the assistant or the transcript, tells it
what to say, or tells it to drop its rules) is not evidence of anything but the order. It errs both ways, and
`experiments/e3_team.py` reports how often on dev turns and on ordinary phrases, apart for those the patterns were
written from and those they were not."""
import re

from contextlens.schemas import Turn

_SENTENCE = r"(?:^|[.!?:;,]\s+|\n)(?:please\s+|now\s+|just\s+)?"
# A name stands alone: it ends the clause or is followed by what the speaker wants said, not by a noun ("assistant manager").
_END = r"(?=\s*(?:[,.!?;:]|$)|\s+(?:that|to|about|what|how|the|it|this|you|and)\b)"
_VERB = r"\b(?:ignore|disregard|forget|override|bypass)\s+"
_PATTERNS = {
    "names the assistant": r"^\W*(?:hey |hi |dear |ok |okay |attention )?(?:assistant|ai|chatbot|bot)\b\s*[,:!]"
                           r"|\b(?:to|for) (?:the )?(?:ai|assistant|chatbot)\b\s*[:,!]"
                           r"|\bas (?:the|an?) (?:ai|assistant)\b(?=\s*(?:,|you\b|i\b))"
                           rf"|\byou are (?:now )?(?:an? )?(?:ai|assistant)\b{_END}"
                           r"|\byou(?:'re| are) now in (?:developer|debug|admin|god|jailbreak|unrestricted) mode\b",
    # An order to drop what came before, or the transcript itself. Ignoring an email, or the rules of someone's league,
    # is ordinary talk and is not aimed at the assistant.
    "drops the rules": rf"{_VERB}(?:(?:all|any|every|the|your|these)\s+)*(?:previous|prior|earlier|preceding|system|original)"
                       r"\s+(?:instructions?|rules?|prompts?|messages?|context|guidelines?|directions?|conversation|text)\b"
                       rf"|{_VERB}(?:the\s+)?above\b"
                       rf"|{_VERB}(?:the|this|your|all)\s+(?:transcript|conversation|question|instructions?|rules?|guidelines?|prompt)"
                       r"\s+and\s+(?:tell|say|answer|reply|print|write|respond|instead)\b"
                       r"|\bstop following (?:your|the|these) (?:rules|instructions|guidelines)\b",
    "says what to say": rf"\b(?:tell|inform)\s+(?:the\s+)?(?:user|assistant)\b{_END}"
                        rf"|{_SENTENCE}claim\s+that\b"
                        rf"|{_SENTENCE}(?:reply|respond|answer|print)\s+(?:only\s+)?(?:with\s+)?['\"]",
    "asks for secrets": r"\breveal\b[^.!?\n]{0,40}\b(?:hidden|system|secret|private|your|all|any)\b[^.!?\n]{0,25}"
                        r"\b(?:prompt|keys?|codes?|notes?|instructions?|secrets?|passwords?)\b"
                        r"|\byour (?:hidden|system|secret) (?:prompt|instructions|rules)\b",
    # A short order to defy the assistant's own setup. It is addressed with "your", so "disobey the speed limit" is not it.
    "bare order": rf"{_SENTENCE}(?:disobey|defy|circumvent|overrule|sidestep|abandon|set aside)\s+(?:all\s+|any\s+|every\s+)?"
                  r"your\s+(?:own\s+)?(?:setup|set-up|configuration|programming|training|settings|safeguards?|guardrails?|"
                  r"restrictions?|constraints?|limits|limitations|rules|instructions|guidelines|filters|directives)\b",
    "markup or commands": r"<\s*/?\s*(?:instructions?|system|assistant)\b|\brm\s+-rf\b"
                          r"|\bdo not cite (?:any|the) (?:evidence|sources|quotes)\b",
}
_COMPILED = {name: re.compile(rx, re.I) for name, rx in _PATTERNS.items()}


def instruction_kind(text: str) -> str | None:
    """Why `text` reads as an order to the assistant, or None when it does not."""
    return next((name for name, rx in _COMPILED.items() if rx.search(text)), None)


def is_instruction(text: str) -> bool:
    return instruction_kind(text) is not None


def without_instruction_turns(earlier: list[Turn]) -> list[Turn]:
    """The context a consistency model is given: earlier turns that read as orders are left out. It takes only the
    earlier turns, so the statement being scored is never removed, even when it is itself an order."""
    return [t for t in earlier if not is_instruction(t.text)]
