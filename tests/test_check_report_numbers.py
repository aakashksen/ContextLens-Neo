"""tools/check_report_numbers.py: a number in the report or the bullets must come from a claim's run record."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import check_report_numbers as crn  # noqa: E402


def _repo(tmp_path: Path, report: str, bullets: str = "No numbers here.\n", claims: list | None = None) -> Path:
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "results" / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "REPORT.md").write_text(report, encoding="utf-8")
    (tmp_path / "docs" / "RESUME-TARGETS.md").write_text(bullets, encoding="utf-8")
    (tmp_path / "README.md").write_text("Readme.\n", encoding="utf-8")
    rec = {"run_id": "E8-20261107T073426291017Z", "n": 136, "config": {"benchmarks": {"qmsum": {"cases": 367}}}}
    (tmp_path / "results" / "runs" / "E8-20261107T073426291017Z.json").write_text(json.dumps(rec), encoding="utf-8")
    default = [
        {"id": "a", "run_id": rec["run_id"], "metric": "x", "value": 0.8125},
        {"id": "b", "run_id": rec["run_id"], "metric": "latency", "value": 4814.0},
        {"id": "c", "run_id": rec["run_id"], "kind": "interval", "metric": "x", "value": [0.4363, 0.5155]},
        {"id": "d", "run_id": rec["run_id"], "kind": "comparison", "metric": "t", "field": "diff", "value": -794.647},
    ]
    (tmp_path / "results" / "claims.json").write_text(json.dumps({"claims": default if claims is None else claims}), encoding="utf-8")
    return tmp_path


def _problems(tmp_path: Path, report: str, **kw) -> list[str]:
    return crn.check(_repo(tmp_path, report, **kw))


def test_a_clean_repo_passes(tmp_path: Path) -> None:
    assert _problems(tmp_path, "The rate was 0.8125, or 81%, at 4,814 ms (4.8 s) on 136 cases.\n") == []


def test_a_number_not_in_a_record_fails(tmp_path: Path) -> None:
    out = _problems(tmp_path, "The rate was 0.83 here.\n")
    assert len(out) == 1 and "0.83" in out[0]


def test_numbers_match_at_the_printed_precision_only(tmp_path: Path) -> None:
    assert _problems(tmp_path, "About 0.81 and 81% and 0.8 and 5 s.\n") == []  # 0.8125 rounds to 0.81, 0.8; 4.814 s to 5
    out = _problems(tmp_path, "About 0.82 and 82% and 0.9.\n")
    assert len(out) == 3


def test_half_values_round_both_ways(tmp_path: Path) -> None:
    claims = [{"id": "a", "run_id": "E8-20261107T073426291017Z", "metric": "x", "value": 0.125}]
    assert _problems(tmp_path, "0.12 and 0.13.\n", claims=claims) == []
    assert len(_problems(tmp_path, "0.14.\n", claims=claims)) == 1


def test_a_percent_is_the_fraction_times_a_hundred_and_only_that(tmp_path: Path) -> None:
    assert _problems(tmp_path, "81% and 81.3% and 81 percent.\n") == []
    assert len(_problems(tmp_path, "0.8125%.\n")) == 1
    assert len(_problems(tmp_path, "4814%.\n")) == 1


def test_milliseconds_may_be_shown_as_seconds(tmp_path: Path) -> None:
    assert _problems(tmp_path, "4.814 s and 4.8 s.\n") == []
    assert len(_problems(tmp_path, "4.9 s.\n")) == 1


def test_a_difference_may_be_shown_without_its_sign_and_list_values_are_read(tmp_path: Path) -> None:
    assert _problems(tmp_path, "Worse by 795 tokens, interval 0.4363 to 0.5155.\n") == []


def test_case_counts_of_a_cited_record_pass(tmp_path: Path) -> None:
    assert _problems(tmp_path, "136 cases, 367 questions.\n") == []
    assert len(_problems(tmp_path, "137 cases.\n")) == 1


def test_a_record_no_claim_cites_gives_no_counts(tmp_path: Path) -> None:
    other = {"run_id": "E5-1", "n": 148, "config": {}}
    repo = _repo(tmp_path, "148 cases.\n")
    (repo / "results" / "runs" / "E5-1.json").write_text(json.dumps(other), encoding="utf-8")
    assert len(crn.check(repo)) == 1


@pytest.mark.parametrize("text", [
    "Run E8-20261107T073426291017Z on 2026-11-07.",
    "See entry 41 and entries 49, 51 and 56, and entry 29a.",
    "Version v3 of Qwen3.5-2B and qwen3.5:4b-q4_K_M, task T9.3, gate H4.",
    "Python 3.12.",
    "AMI is CC BY 4.0.",
    "recall@5 and p95 and ES2003a.",
    "## 3. Data\n\n### 4.2 Detail\n\n1. First\n2) Second\n",
    "```\ndocker run -p 8000:8000 --gpus 1\n```\n",
    "1. Start it\n   ```\n   docker run -p 8000:8000\n   ```\n",
])
def test_the_narrow_ignore_rules(tmp_path: Path, text: str) -> None:
    assert _problems(tmp_path, text + "\n") == []


@pytest.mark.parametrize("text", [
    "Run 20261107 passed.",
    "The 7 prompts.",
    "Entry results of 41 cases.",
    "Python 3.11.",
    "CC BY 4.0 and 4.0 more.",
    "At 8000 ms.",
    "In 2026-11 only.",
])
def test_the_ignore_rules_do_not_hide_other_numbers(tmp_path: Path, text: str) -> None:
    assert len(_problems(tmp_path, text + "\n")) >= 1


def test_bullets_are_checked_too(tmp_path: Path) -> None:
    out = _problems(tmp_path, "Fine.\n", bullets="1. Cut tokens by 77% on 136 cases.\n")
    assert len(out) == 1 and "RESUME-TARGETS.md" in out[0] and "77" in out[0]


@pytest.mark.parametrize("text", ["A [placeholder] here.", "Value [pending: D1].", "Fill [tokens_ratio]x.", "TODO later.", "TBD."])
def test_a_placeholder_fails(tmp_path: Path, text: str) -> None:
    out = _problems(tmp_path, text + "\n")
    assert any("placeholder" in p for p in out)


def test_links_checkboxes_and_code_are_not_placeholders(tmp_path: Path) -> None:
    text = "See [the design](docs/DESIGN.md), `x[0]`, and [x] done.\n\n```\nlist[int]\n```\n"
    assert _problems(tmp_path, text) == []


def test_empty_claims_fail(tmp_path: Path) -> None:
    out = _problems(tmp_path, "Plain.\n", claims=[])
    assert any("claims.json" in p for p in out)


def test_a_claim_with_a_missing_record_fails(tmp_path: Path) -> None:
    out = _problems(tmp_path, "Plain.\n", claims=[{"id": "z", "run_id": "E9-none", "metric": "x", "value": 1.0}])
    assert any("E9-none" in p for p in out)


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path, "Rate 0.8125.\n")
    assert crn.main([str(repo)]) == 0
    assert "ok" in capsys.readouterr().out
    (repo / "docs" / "REPORT.md").write_text("Rate 0.99.\n", encoding="utf-8")
    assert crn.main([str(repo)]) == 1
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["It was 9.9x faster.\n", "It took 2400ms.\n", "It took 7.7s.\n"])
def test_a_number_glued_to_a_unit_is_still_checked(tmp_path: Path, text: str) -> None:
    out = _problems(tmp_path, text)
    assert len(out) == 1 and "is in no cited run record" in out[0]


def test_a_supported_number_glued_to_a_unit_passes(tmp_path: Path) -> None:
    assert _problems(tmp_path, "It took 4814ms, or 4.8s, on 136 cases at the 95th percentile.\n") == []
