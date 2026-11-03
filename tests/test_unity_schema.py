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


def test_extra_field_with_trailing_comment_fails(tmp_path):
    errs = _broken(tmp_path, "        public int tokens;\n", "        public int tokens;\n        public int zzz; // secret\n")
    assert "Card.zzz: extra in C# (not in schema)" in errs


def test_hidden_serialized_field_fails(tmp_path):
    for decl in ("[SerializeField] int zzz;", "[SerializeField] private int zzz;", "private int zzz;", "int zzz;"):
        errs = _broken(tmp_path / str(abs(hash(decl))), "        public int tokens;\n", f"        public int tokens;\n        {decl}\n")
        assert any("Card" in e and "cannot read" in e for e in errs), decl


def test_two_fields_on_one_line_fail(tmp_path):
    errs = _broken(tmp_path, "public int tokens;", "public int tokens; public int hidden;")
    assert any("cannot read" in e for e in errs)


def test_property_instead_of_field_fails(tmp_path):
    errs = _broken(tmp_path, "public int tokens;", "public int tokens { get; set; }")
    assert any("cannot read" in e for e in errs)


def test_one_line_class_fails(tmp_path):
    old = '    public class PauseMessage\n    {\n        public string type = "pause";\n    }'
    errs = _broken(tmp_path, old, '    public class PauseMessage { public string type = "pause"; public int x; }')
    assert any("cannot read" in e for e in errs)


def test_serializable_class_outside_messages_fails(tmp_path):
    shutil.copytree(mod.CS_DIR, tmp_path / "s")
    (tmp_path / "s" / "Other.cs").write_text("[Serializable]\npublic class Sneaky\n{\n    public int a;\n}\n", encoding="utf-8")
    errs = mod.check(tmp_path / "s")
    assert any(e.startswith("Other.cs") and "Messages.cs" in e for e in errs)


def test_missing_messages_file_fails(tmp_path):
    (tmp_path / "s").mkdir()
    assert mod.check(tmp_path / "s")


def test_type_constant_changed_fails(tmp_path):
    errs = _broken(tmp_path, 'type = "press"', 'type = "prss"')
    assert any(e.startswith("PressMessage.type: default should be") for e in errs)


def test_type_constant_removed_fails(tmp_path):
    errs = _broken(tmp_path, 'public string type = "press";', "public string type;")
    assert any(e.startswith("PressMessage.type: default should be") for e in errs)


def test_non_const_default_changed_fails(tmp_path):
    errs = _broken(tmp_path, "public int sample_rate = 16000;", "public int sample_rate = 8000;")
    assert any(e.startswith("AudioMessage.sample_rate: default should be 16000") for e in errs)


def test_non_const_default_removed_fails(tmp_path):
    errs = _broken(tmp_path, 'public string detail = "standard";', "public string detail;")
    assert any(e.startswith("ButtonRequest.detail: default should be") for e in errs)


def test_default_on_required_field_fails(tmp_path):
    errs = _broken(tmp_path, "public string data;", 'public string data = "x";')
    assert any(e.startswith("AudioMessage.data: required in schema") for e in errs)


def test_nullable_marker_removed_fails(tmp_path):
    errs = _broken(tmp_path, "public string suggestion; // nullable", "public string suggestion;")
    assert "Card.suggestion: nullable in schema but not marked `nullable` in C#" in errs


def test_nullable_marker_on_non_nullable_fails(tmp_path):
    errs = _broken(tmp_path, "public string quote;", "public string quote; // nullable")
    assert "EvidenceRef.quote: marked `nullable` in C# but not nullable in schema" in errs


def test_envelope_extra_field_fails(tmp_path):
    errs = _broken(tmp_path, "        public string type;\n    }", "        public string type;\n        public int q;\n    }")
    assert "Envelope: must have exactly the field `type`" in errs
