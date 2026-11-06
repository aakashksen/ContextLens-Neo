"""Check 3: does Ollama serve the memory and consistency requests one at a time? Fakes only: no server is called."""
import json
import threading
import time
from pathlib import Path

import pytest

from contextlens.experiments import ollama_concurrency as oc


class Server:
    """A fake model server: each request takes `work_s`; with `slots` of 1 the requests queue."""

    def __init__(self, work_s: float, slots: int) -> None:
        self.work_s, self.gate = work_s, threading.Semaphore(slots)

    def __call__(self, name: str) -> None:
        with self.gate:
            time.sleep(self.work_s)


def test_two_requests_on_one_slot_take_the_sum_and_the_verdict_says_one_at_a_time():
    server = Server(0.05, 1)
    res = oc.measure({"memory": lambda: server("memory"), "consistency": lambda: server("consistency")}, repeats=2)
    assert res["verdict"] == "one at a time" and res["both_over_sum_of_alone"] > 0.85


def test_two_requests_on_two_slots_overlap_and_the_verdict_says_parallel():
    server = Server(0.05, 2)
    res = oc.measure({"memory": lambda: server("memory"), "consistency": lambda: server("consistency")}, repeats=2)
    assert res["verdict"] == "in parallel" and res["both_over_sum_of_alone"] < 0.7
    assert set(res["alone_s"]) == {"memory", "consistency"} and len(res["both_wall_s"]) == 2


def test_the_result_file_goes_to_results_local(tmp_path: Path):
    path = oc.write_result({"verdict": "in parallel"}, {"version": "0.34.1"}, tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent == tmp_path / "results/local" and path.name.startswith("ollama-concurrency-")
    assert doc["diagnostic"] is True and doc["server"] == {"version": "0.34.1"} and doc["verdict"] == "in parallel"


def test_an_unusable_measurement_is_refused():
    with pytest.raises(ValueError, match="repeats"):
        oc.measure({"a": lambda: None, "b": lambda: None}, repeats=0)
