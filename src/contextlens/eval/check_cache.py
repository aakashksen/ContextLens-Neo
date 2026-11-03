"""Remember which run records already passed check_run, so a land rechecks only what is new or changed.

A pass is reused only when nothing it depended on has changed: the record file, its predictions files, the
scenario file and manifests it was checked against, the `--min-n` and `--split` of the check, and the source of
check_run and every contextlens module it imports (so any code change forces a full recheck). A record that
failed is never stored."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from contextlens.eval.records import SCENARIO_SOURCES, SUBSETS_REL, write_atomic

SRC = Path(__file__).resolve().parents[2]
ENTRY = "contextlens.eval.check_run"
CACHE_REL = Path("results/.check-cache.json")


def _module_file(name: str) -> Path | None:
    base = SRC.joinpath(*name.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return {m for m in found if m.startswith("contextlens")}


def code_modules(entry: str = ENTRY) -> dict[str, Path]:
    """Every contextlens module the entry imports, directly or not, with its source file (package
    __init__ files included)."""
    todo, seen = [entry], {}
    while todo:
        name = todo.pop()
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            path = _module_file(sub)
            if path is not None and sub not in seen:
                seen[sub] = path
                todo += sorted(_imports(path))
    return seen


def code_hash() -> str:
    h = hashlib.sha256(f"python{sys.version_info[0]}.{sys.version_info[1]}".encode())
    for name, path in sorted(code_modules().items()):
        h.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


def record_key(path: Path, rec: dict[str, Any], root: Path, min_n: int, split: str | None, code: str) -> str:
    """Hash of everything check_record reads for this record, plus the code and the check's own arguments."""
    deps: list[str] = [str(min_n), split or "", code, _file_hash(path)]
    preds = rec.get("predictions")
    for rel in sorted(preds.values()) if isinstance(preds, dict) else []:
        deps.append(_file_hash(root / rel) if isinstance(rel, str) else "bad")
    source = rec.get("scenario_source")
    if source in SCENARIO_SOURCES:
        manifest_rel, directory = SCENARIO_SOURCES[source]
        deps.append(_file_hash(root / manifest_rel))
        deps.append(_file_hash(root / directory / f"{rec.get('split')}.jsonl"))
    subset = rec.get("subset")
    if isinstance(subset, str):
        deps.append(_file_hash(root / SUBSETS_REL / f"{subset}.json"))
    return hashlib.sha256("\n".join(deps).encode()).hexdigest()


class PassCache:
    def __init__(self, root: Path, code: str, *, read: bool = True) -> None:
        self.path = root / CACHE_REL
        self.code = code
        self.passed: dict[str, str] = {}
        if read and self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("code") == code and isinstance(data.get("passed"), dict):
                    self.passed = data["passed"]
            except ValueError:
                pass  # an unreadable cache is an empty one

    def hit(self, key: str) -> bool:
        return key in self.passed

    def add(self, key: str, name: str) -> None:
        self.passed[key] = name

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(self.path, json.dumps({"code": self.code, "passed": self.passed}, indent=1) + "\n")
