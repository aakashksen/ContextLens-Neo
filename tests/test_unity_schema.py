import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_unity_schema", ROOT / "tools" / "check_unity_schema.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _broken(tmp_path, old: str, new: str) -> list[str]:
    shutil.copytree(mod.CS_DIR, tmp_path / "s")
    f = tmp_path / "s" / "Messages.cs"
    text = f.read_text(encoding="utf-8")
    assert old in text
    f.write_text(text.replace(old, new, 1), encoding="utf-8")
    return mod.check(tmp_path / "s")


def test_real_client_matches_schema():
    assert mod.check() == []


def test_missing_field_fails(tmp_path):
    errs = _broken(tmp_path, "        public int as_of_seq;\n", "")
    assert "ButtonRequest.as_of_seq: missing in C#" in errs


def test_extra_field_fails(tmp_path):
    errs = _broken(tmp_path, "        public int tokens;\n", "        public int tokens;\n        public int bonus;\n")
    assert "Card.bonus: extra in C# (not in schema)" in errs


def test_wrong_type_fails(tmp_path):
    errs = _broken(tmp_path, "public int seq;", "public string seq;")
    assert any(e.startswith("Turn.seq: schema type integer") for e in errs)


def test_wrong_enum_fails(tmp_path):
    errs = _broken(tmp_path, "enum: recording|paused|deleted|error", "enum: recording|paused")
    assert any(e.startswith("StatusMessage.state: enum") for e in errs)
