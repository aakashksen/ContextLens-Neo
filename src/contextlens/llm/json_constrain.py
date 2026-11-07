"""Schema-constrained decoding for the local Hugging Face model.

Ollama is given the card's JSON schema as `format`, so its output is valid by construction. A local model run through
`generate` has no such thing, which made E5's schema-valid rate measure something E6's never did. `prefix_status` says
whether a text can still become a value of a JSON schema, and `SchemaLogitsProcessor` lets only such tokens through.
It reads the subset of JSON Schema that pydantic writes for the card: objects, arrays, strings, numbers, booleans, null,
enums, and anyOf. Nothing here is imported until a constrained call is made."""
from __future__ import annotations

import json
import re
from typing import Any

INVALID, INCOMPLETE, COMPLETE = "invalid", "incomplete", "complete"
_OK, _INC, _BAD = "ok", "inc", "bad"
_WS = " \t\n\r"
_HEX = "0123456789abcdefABCDEF"
_NUM = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?")
_INT = re.compile(r"-?(0|[1-9]\d*)")
_NUM_PART = re.compile(r"-?\d*(\.\d*)?([eE][+-]?\d*)?")
_INT_PART = re.compile(r"-?\d*")


def _resolve(node: Any, root: dict) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        node = root["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    return node


def _ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in _WS:
        i += 1
    return i


def _literal(s: str, i: int, options: list[str]) -> tuple[str, int]:
    rest = s[i:]
    inc = False
    for c in options:
        if rest.startswith(c):
            return _OK, i + len(c)
        inc = inc or c.startswith(rest)
    return (_INC if inc else _BAD), i


def _string(s: str, i: int) -> tuple[str, int]:
    """A JSON string starting at s[i] == '"'. Ends at the closing quote."""
    j = i + 1
    while j < len(s):
        ch = s[j]
        if ch == '"':
            return _OK, j + 1
        if ord(ch) < 0x20:
            return _BAD, i
        if ch == "\\":
            if j + 1 >= len(s):
                return _INC, i
            nxt = s[j + 1]
            if nxt == "u":
                hexs = s[j + 2:j + 6]
                if not all(h in _HEX for h in hexs):
                    return _BAD, i
                if len(hexs) < 4:
                    return _INC, i
                j += 6
                continue
            if nxt not in '"\\/bfnrt':
                return _BAD, i
            j += 2
            continue
        j += 1
    return _INC, i


def _object(schema: dict, root: dict, s: str, i: int) -> tuple[str, int]:
    props, required = schema.get("properties", {}), set(schema.get("required", []))
    i += 1
    seen: set[str] = set()
    first = True
    while True:
        i = _ws(s, i)
        if i >= len(s):
            return _INC, i
        if s[i] == "}":
            return (_OK, i + 1) if required <= seen else (_BAD, i)
        if not first:
            if s[i] != ",":
                return _BAD, i
            i = _ws(s, i + 1)
            if i >= len(s):
                return _INC, i
        if s[i] != '"':
            return _BAD, i
        st, j = _string(s, i)
        if st == _INC:
            partial = s[i + 1:]
            return (_INC if any(p.startswith(partial) for p in props if p not in seen) else _BAD), i
        if st == _BAD:
            return _BAD, i
        key = json.loads(s[i:j])
        if key not in props or key in seen:
            return _BAD, i
        seen.add(key)
        i = _ws(s, j)
        if i >= len(s):
            return _INC, i
        if s[i] != ":":
            return _BAD, i
        st, i = _value(props[key], root, s, i + 1)
        if st != _OK:
            return st, i
        first = False


def _array(schema: dict, root: dict, s: str, i: int) -> tuple[str, int]:
    item = schema.get("items", {})
    i = _ws(s, i + 1)
    if i >= len(s):
        return _INC, i
    if s[i] == "]":
        return _OK, i + 1
    while True:
        st, i = _value(item, root, s, i)
        if st != _OK:
            return st, i
        i = _ws(s, i)
        if i >= len(s):
            return _INC, i
        if s[i] == "]":
            return _OK, i + 1
        if s[i] != ",":
            return _BAD, i
        i += 1


def _value(schema: Any, root: dict, s: str, i: int) -> tuple[str, int]:
    schema = _resolve(schema, root)
    i = _ws(s, i)
    if i >= len(s):
        return _INC, i
    if "anyOf" in schema:
        best, inc = None, False
        for alt in schema["anyOf"]:
            st, j = _value(alt, root, s, i)
            if st == _OK and best is None:
                best = j
            inc = inc or st == _INC
        if inc:
            return _INC, i
        return (_OK, best) if best is not None else (_BAD, i)
    if "const" in schema or "enum" in schema:
        return _literal(s, i, [json.dumps(v) for v in (schema["enum"] if "enum" in schema else [schema["const"]])])
    kind = schema.get("type")
    if kind == "object":
        return _object(schema, root, s, i) if s[i] == "{" else (_BAD, i)
    if kind == "array":
        return _array(schema, root, s, i) if s[i] == "[" else (_BAD, i)
    if kind == "string":
        return _string(s, i) if s[i] == '"' else (_BAD, i)
    if kind in ("integer", "number"):
        rest = s[i:]
        if (_NUM_PART if kind == "number" else _INT_PART).fullmatch(rest):
            return _INC, i  # the text ends inside the number, which may still grow
        m = (_INT if kind == "integer" else _NUM).match(rest)
        return (_OK, i + m.end()) if m else (_BAD, i)
    if kind == "boolean":
        return _literal(s, i, ["true", "false"])
    if kind == "null":
        return _literal(s, i, ["null"])
    return _BAD, i


def prefix_status(schema: dict, text: str) -> str:
    """INVALID when no continuation of `text` is a value of `schema`; COMPLETE when it already is one (only whitespace
    may follow); otherwise INCOMPLETE."""
    st, j = _value(schema, schema, text, 0)
    if st == _BAD:
        return INVALID
    if st == _INC:
        return INCOMPLETE
    return COMPLETE if not text[j:].strip(_WS) else INVALID


class SchemaLogitsProcessor:
    """For one sequence: keeps, among the likeliest tokens, those after which the text can still become a value of the
    schema, and nothing but the end token once it is one. `prompt_len` is the length of the prompt in ids."""

    def __init__(self, tok: Any, schema: dict, prompt_len: int, top_n: int = 50, eos_ids: list[int] | None = None) -> None:
        self.tok, self.schema, self.prompt_len, self.top_n = tok, schema, prompt_len, top_n
        self.eos = list(eos_ids if eos_ids is not None else [tok.eos_token_id])

    def _ok(self, ids: list[int], cand: int) -> bool:
        return cand not in self.eos and prefix_status(self.schema, self.tok.decode(ids + [cand])) != INVALID

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        ids = input_ids[0, self.prompt_len:].tolist()
        out = torch.full_like(scores, float("-inf"))
        if prefix_status(self.schema, self.tok.decode(ids)) == COMPLETE:
            out[:, self.eos] = scores[:, self.eos]
            return out
        order = torch.argsort(scores[0], descending=True).tolist()
        n_tok = min(len(self.tok), scores.shape[-1])
        valid: list[int] = []
        for start in range(0, len(order), self.top_n):  # the likeliest window first; later ones only if it had none
            valid = [c for c in order[start:start + self.top_n] if c < n_tok and self._ok(ids, c)]
            if valid:
                break
        if not valid:
            return scores
        out[0, valid] = scores[0, valid]
        return out
