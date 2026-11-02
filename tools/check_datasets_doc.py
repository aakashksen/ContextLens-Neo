"""Fail unless each required dataset section in docs/datasets.md has the required lines."""
import re
import sys
from pathlib import Path

DATASETS = ["DECODE", "QMSum", "LongMemEval", "AMI Meeting Corpus"]
FIELDS = ["URL", "License", "Labels", "Approved"]


def sections(text: str) -> dict[str, str]:
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    return {p.split("\n", 1)[0].strip(): p for p in parts}


def check(text: str) -> list[str]:
    secs = sections(text)
    missing = []
    for name in DATASETS:
        body = secs.get(name)
        if body is None:
            missing.append(f"{name}: section missing")
            continue
        for field in FIELDS:
            if not re.search(rf"^{field}:[ \t]*\S", body, flags=re.M):
                missing.append(f"{name}: {field} line missing")
    return missing


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "datasets.md"
    missing = check(path.read_text(encoding="utf-8"))
    for m in missing:
        print(m, file=sys.stderr)
    if missing:
        return 1
    print("datasets doc ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
