"""Check the Unity client's C# message classes against docs/client-contract.schema.json.

Standard library only. Usage: python tools/check_unity_schema.py [cs_dir]
Exit 0 when every schema type has a C# class with exactly the same fields, types, enum values,
nullable markers and defaults (including the `type` constants), and nothing in Messages.cs is left
unread. Exit 1 with the problems otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "docs" / "client-contract.schema.json"
CS_DIR = ROOT / "clients" / "unity" / "Assets" / "Scripts"
NOT_IN_CONTRACT = {"Envelope"}  # helper classes allowed in the C# files

WIRE_FILE = "Messages.cs"
TYPE_RE = r"\w+(?:<\s*\w+\s*>)?(?:\[\])?"
FIELD_RE = re.compile(rf"^public\s+({TYPE_RE})\s+(\w+)\s*(?:=\s*(.+?))?\s*;$")
CLASS_LINE_RE = re.compile(r"^public\s+class\s+(\w+)$")
PRIMS = {"string": "string", "int": "integer", "long": "integer", "float": "number",
         "double": "number", "bool": "boolean"}


def cs_type(t: str) -> str:
    t = t.strip()
    m = re.fullmatch(r"List<\s*(.+?)\s*>", t)
    if m:
        return f"array<{cs_type(m.group(1))}>"
    if t.endswith("[]"):
        return f"array<{cs_type(t[:-2])}>"
    return PRIMS.get(t, f"ref:{t}")


def split_comment(line: str) -> tuple[str, str]:
    """Split a line into (code, comment) at the first `//` outside a string literal."""
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_str = not in_str
        elif not in_str and line.startswith("//", i):
            return line[:i], line[i + 2:]
    return line, ""


def _blank_lines(m: re.Match) -> str:
    return "\n" * m.group(0).count("\n")  # keep line numbers when dropping a block comment


def parse_cs(cs_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """Read the wire classes. Anything not understood is returned as a problem, never skipped."""
    problems: list[str] = []
    wire = cs_dir / WIRE_FILE
    for f in sorted(cs_dir.glob("*.cs")):
        if f.name != WIRE_FILE and "[Serializable]" in f.read_text(encoding="utf-8"):
            problems.append(f"{f.name}: [Serializable] types must live in {WIRE_FILE}")
    if not wire.is_file():
        return {}, problems + [f"{WIRE_FILE}: not found in {cs_dir}"]
    text = re.sub(r"/\*.*?\*/", _blank_lines, wire.read_text(encoding="utf-8"), flags=re.S)
    classes: dict[str, dict] = {}
    current: str | None = None  # class whose body we are inside
    pending: str | None = None  # class declared, waiting for its "{"
    depth = 0
    for n, raw in enumerate(text.splitlines(), 1):
        code, comment = split_comment(raw)
        code = code.strip()
        if not code:
            continue
        where = f"{WIRE_FILE}:{n}"
        if code == "{":
            depth += 1
            if pending:
                current, pending = pending, None
        elif code == "}":
            depth -= 1
            if current is not None and depth == 1:
                current = None
        elif pending:
            problems.append(f"{where}: cannot read this line (expected the class's opening brace): {code}")
        elif current is not None:
            m = FIELD_RE.match(code)
            if not m:
                problems.append(f"{current}: cannot read this line ({where}): {code}")
                continue
            ftype, fname, default = m.groups()
            em = re.search(r"\benum:\s*([\w|]+)", comment)
            if fname in classes[current]:
                problems.append(f"{current}.{fname}: declared twice")
            classes[current][fname] = (cs_type(ftype), default, em.group(1) if em else None,
                                       bool(re.search(r"\bnullable\b", comment)))
        elif depth == 1 and (m := CLASS_LINE_RE.match(code)):
            if m.group(1) in classes:
                problems.append(f"{m.group(1)}: class declared twice")
            classes.setdefault(m.group(1), {})
            pending = m.group(1)
        elif depth == 1 and code == "[Serializable]":
            pass
        elif depth == 0 and (code.startswith("using ") or re.fullmatch(r"namespace\s+[\w.]+", code)):
            pass
        else:
            problems.append(f"{where}: cannot read this line: {code}")
    if depth != 0 or pending or current:
        problems.append(f"{WIRE_FILE}: unbalanced braces")
    return classes, problems


def schema_type(prop: dict) -> str:
    if "anyOf" in prop:
        options = [o for o in prop["anyOf"] if o.get("type") != "null"]
        return schema_type(options[0]) if len(options) == 1 else "?"
    if "$ref" in prop:
        return "ref:" + prop["$ref"].rsplit("/", 1)[1]
    t = prop.get("type")
    if t == "array":
        return f"array<{schema_type(prop['items'])}>"
    return t or "?"


def schema_enum(prop: dict) -> list[str] | None:
    if "anyOf" in prop:
        for o in prop["anyOf"]:
            if "enum" in o:
                return o["enum"]
    return prop.get("enum")


def load_schema_defs(path: Path) -> dict[str, dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    defs: dict[str, dict] = {}
    for side in (doc["client_to_server"], doc["server_to_client"]):
        defs.update(side.get("$defs", {}))
    return defs


def cs_literal(text: str | None):
    """Value of a simple C# initializer, or a sentinel when it is not a plain literal."""
    t = (text or "").strip()
    if re.fullmatch(r"new\s+List<\s*\w+\s*>\s*\(\s*\)", t):
        return []
    if t == "null":
        return None
    if t in ("true", "false"):
        return t == "true"
    if t.startswith('"'):
        try:
            return json.loads(t)
        except ValueError:
            return _BAD
    try:
        return float(t.rstrip("fFdD"))
    except ValueError:
        return _BAD


_BAD = object()


def zero_value(prop: dict, nullable: bool):
    if nullable:
        return None
    return {"integer": 0, "number": 0.0, "boolean": False, "string": "", "array": []}.get(prop.get("type"), _BAD)


def check_defaults(name: str, p: str, prop: dict, required: bool, init: str | None, nullable: bool) -> list[str]:
    if required:
        if init is not None:
            return [f"{name}.{p}: required in schema but C# gives it a default ({init})"]
        return []
    want = prop["default"]
    got = cs_literal(init) if init is not None else zero_value(prop, nullable)
    if init is None and want in ("", None) and prop.get("type") == "string":
        got = want  # an uninitialised C# string / list reads as the empty value
    if got is _BAD or got != want or (isinstance(want, bool) != isinstance(got, bool)):
        return [f"{name}.{p}: default should be {json.dumps(want)}, got {init if init is not None else '(none)'}"]
    return []


def check(cs_dir: Path = CS_DIR, schema: Path = SCHEMA) -> list[str]:
    defs = load_schema_defs(schema)
    classes, errors = parse_cs(cs_dir)
    for name, d in defs.items():
        if name not in classes:
            errors.append(f"{name}: missing C# class")
            continue
        props, got = d["properties"], classes[name]
        for p in sorted(props.keys() - got.keys()):
            errors.append(f"{name}.{p}: missing in C#")
        for p in sorted(got.keys() - props.keys()):
            errors.append(f"{name}.{p}: extra in C# (not in schema)")
        for p in sorted(props.keys() & got.keys()):
            prop = props[p]
            want, (have, init, enum, nullable) = schema_type(prop), got[p]
            if want != have:
                errors.append(f"{name}.{p}: schema type {want}, C# type {have}")
            allowed = schema_enum(prop)
            if allowed is not None and (enum is None or sorted(enum.split("|")) != sorted(allowed)):
                errors.append(f"{name}.{p}: enum should be {'|'.join(allowed)}, C# says {enum}")
            is_nullable = "anyOf" in prop and any(o.get("type") == "null" for o in prop["anyOf"])
            if is_nullable and not nullable:
                errors.append(f"{name}.{p}: nullable in schema but not marked `nullable` in C#")
            if nullable and not is_nullable:
                errors.append(f"{name}.{p}: marked `nullable` in C# but not nullable in schema")
            required = p in d.get("required", [])
            errors += check_defaults(name, p, prop, required, init, is_nullable)
    if "Envelope" in classes and set(classes["Envelope"]) != {"type"}:
        errors.append("Envelope: must have exactly the field `type`")
    for name in sorted(classes.keys() - defs.keys() - NOT_IN_CONTRACT):
        errors.append(f"{name}: C# class not in schema")
    return errors


def main(argv: list[str]) -> int:
    cs_dir = Path(argv[1]) if len(argv) > 1 else CS_DIR
    errors = check(cs_dir)
    if errors:
        print("Unity schema check FAILED:")
        print("\n".join("  " + e for e in errors))
        return 1
    print(f"Unity schema check OK ({len(load_schema_defs(SCHEMA))} types match)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
