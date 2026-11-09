"""Fail if a number in the report, README or bullets is not in a run record a claim cites, if a [placeholder]
remains, or if results/claims.json is empty. The claims themselves are checked by `python -m contextlens.eval.claims check`."""
import json
import re
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ("docs/REPORT.md", "docs/RESUME-TARGETS.md", "README.md")

FENCE = re.compile(r"^```.*?^```[ \t]*$", re.M | re.S)
INLINE = re.compile(r"`[^`\n]*`")
# The only things not read as numbers. Each is narrow on purpose; see tests/test_check_report_numbers.py.
IGNORED = [
    re.compile(r"\b[A-Z][A-Z0-9]*-\d{8}T\d+Z\b"),  # run ids
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # dates
    re.compile(r"\bentr(?:y|ies) \d+[a-z]?(?:(?:,| and| to|-) ?\d+[a-z]?)*", re.I),  # "entry 41", "entries 49, 51 and 56"
    re.compile(r"\bPython 3\.12\b"),
    re.compile(r"@\d+"),  # recall@5
    re.compile(r"^#+[ \t]*\d+(?:\.\d+)*\.?(?=\s)", re.M),  # heading numbers
    re.compile(r"^[ \t]*\d+[.)](?=\s)", re.M),  # ordered list markers
]
NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w.]*\w)(%| percent\b)?")
MODEL_SIZE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?[Bb](?![A-Za-z0-9])")  # 2B, 4b
PLACEHOLDER = re.compile(r"\[[^\]\n]*[A-Za-z_][^\]\n]*\](?![(:])|\bTODO\b|\bTBD\b")


def _text(doc: str) -> str:
    return INLINE.sub("", FENCE.sub("", doc))


def numbers(doc: str) -> list[str]:
    text = FENCE.sub("", doc)
    for rx in IGNORED:
        text = rx.sub(" ", text)
    text = MODEL_SIZE.sub(" ", text)
    return [m.group(0).strip() for m in NUMBER.finditer(text)]


def _cited(root: Path, claims: list[dict]) -> tuple[list[float], set[int], list[str]]:
    values: list[float] = []
    counts: set[int] = set()
    problems: list[str] = []

    def flat(v: object) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            values.append(float(v))
        elif isinstance(v, list):
            for x in v:
                flat(x)

    def cases(o: object) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "cases" and isinstance(v, int) and not isinstance(v, bool):
                    counts.add(v)
                cases(v)
        elif isinstance(o, list):
            for x in o:
                cases(x)

    seen: set[str] = set()
    for c in claims:
        flat(c.get("value"))
        run = c.get("run_id")
        if run in seen:
            continue
        seen.add(run)
        path = root / "results" / "runs" / f"{run}.json"
        if not path.is_file():
            problems.append(f"claim {c.get('id', '?')}: no run record {run}")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rec.get("n"), int):
            counts.add(rec["n"])
        cases(rec)
    return values, counts, problems


def _rounds_to(value: float, printed: str) -> bool:
    shown = Decimal(printed)
    exp = shown.as_tuple().exponent
    unit = Decimal(1).scaleb(exp)  # type: ignore[arg-type]
    return any(Decimal(repr(value)).quantize(unit, rounding=mode) == shown for mode in (ROUND_HALF_EVEN, ROUND_HALF_UP))


def _supported(token: str, values: list[float], counts: set[int]) -> bool:
    percent = token.endswith("%") or token.endswith("percent")
    printed = re.sub(r"[^\d.]", "", token.replace(",", ""))
    if "." not in printed and int(printed) in counts and not percent:
        return True
    for v in values:
        v = abs(v)
        forms = [v * 100] if percent else [v, v / 1000]
        if any(_rounds_to(f, printed) for f in forms):
            return True
    return False


def check(root: Path = ROOT) -> list[str]:
    path = root / "results" / "claims.json"
    claims = json.loads(path.read_text(encoding="utf-8")).get("claims") if path.is_file() else None
    if not claims:
        return ["results/claims.json is empty or missing, so no number can be supported"]
    values, counts, problems = _cited(root, claims)
    for rel in DOCS:
        doc = root / rel
        if not doc.is_file():
            problems.append(f"{rel}: missing")
            continue
        text = doc.read_text(encoding="utf-8")
        problems += [f"{rel}: placeholder {m.group(0)!r} remains" for m in PLACEHOLDER.finditer(_text(text)) if not re.fullmatch(r"\[[ xX]\]", m.group(0))]
        problems += [f"{rel}: number {t!r} is in no cited run record" for t in numbers(text) if not _supported(t, values, counts)]
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    problems = check(Path(args[0]) if args else ROOT)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("report numbers ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
