"""Check the Unity client's C# message classes against docs/client-contract.schema.json.

Standard library only. Usage: python tools/check_unity_schema.py [cs_dir]
Exit 0 when every schema type has a C# class with exactly the same fields and
types (and enum values / `type` constants); exit 1 with the differences otherwise.
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

CLASS_RE = re.compile(r"public\s+class\s+(\w+)\s*\{(.*?)^\s{4}\}", re.S | re.M)
FIELD_RE = re.compile(
    r"^\s*public\s+([\w<>\[\],\s]+?)\s+(\w+)\s*(?:=\s*([^;]+?))?\s*;\s*(?://\s*enum:\s*([\w|]+))?\s*$"
)
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


def parse_cs(cs_dir: Path) -> dict[str, dict[str, tuple[str, str | None, str | None]]]:
    classes: dict[str, dict] = {}
    for f in sorted(cs_dir.glob("*.cs")):
        text = re.sub(r"/\*.*?\*/", "", f.read_text(encoding="utf-8"), flags=re.S)
        for name, body in CLASS_RE.findall(text):
            fields = {}
            for line in body.splitlines():
                m = FIELD_RE.match(line)
                if m:
                    ftype, fname, default, enum = m.groups()
                    fields[fname] = (cs_type(ftype), default, enum)
            classes[name] = fields
    return classes


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


def check(cs_dir: Path = CS_DIR, schema: Path = SCHEMA) -> list[str]:
    defs = load_schema_defs(schema)
    classes = parse_cs(cs_dir)
    errors: list[str] = []
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
            want, (have, default, enum) = schema_type(props[p]), got[p]
            if want != have:
                errors.append(f"{name}.{p}: schema type {want}, C# type {have}")
            allowed = schema_enum(props[p])
            if allowed is not None and (enum is None or sorted(enum.split("|")) != sorted(allowed)):
                errors.append(f"{name}.{p}: enum should be {'|'.join(allowed)}, C# says {enum}")
            if "const" in props[p] and (default or "").strip() != json.dumps(props[p]["const"]):
                errors.append(f"{name}.{p}: default should be {json.dumps(props[p]['const'])}, got {default}")
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
