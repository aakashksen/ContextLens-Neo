"""Check 3: does Ollama serve the memory and consistency specialists' requests one at a time?

    python -m contextlens.experiments.ollama_concurrency

The team runs its memory and consistency specialists in parallel against one Ollama model. If Ollama answers one request
at a time, "in parallel" saves nothing and the slower specialist's time is added to the other's. This sends the real first
request of each specialist (their system prompts, a real AMI question and its transcript window, the same schema and
token limit) alone, three times each, then both at once, three times, and compares the times. About 2 to 3 minutes on the
GPU. The result goes to results/local/ (git-ignored) as a diagnostic.

Verdict: the time of both together divided by the sum of the two alone. At 0.85 or more they were served one at a time; at
0.7 or less they overlapped; in between they partly overlapped."""

import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.eval.records import ROOT

REPEATS = 3
ONE_AT_A_TIME, PARALLEL = 0.85, 0.7


def measure(calls: dict[str, Callable[[], Any]], repeats: int = REPEATS) -> dict[str, Any]:
    """Time each call alone, then all at once; `repeats` times each."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    alone: dict[str, list[float]] = {n: [] for n in calls}
    for n, fn in calls.items():
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            alone[n].append(time.perf_counter() - t0)
    both_wall, both_each = [], []
    for _ in range(repeats):
        each: dict[str, float] = {}

        def timed(name: str, fn: Callable[[], Any]) -> None:
            t = time.perf_counter()
            fn()
            each[name] = time.perf_counter() - t

        threads = [threading.Thread(target=timed, args=(n, fn)) for n, fn in calls.items()]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        both_wall.append(time.perf_counter() - t0)
        both_each.append(each)
    ratio = statistics.median(both_wall) / sum(statistics.median(v) for v in alone.values())
    verdict = "one at a time" if ratio >= ONE_AT_A_TIME else "in parallel" if ratio <= PARALLEL else "partly in parallel"
    return {"alone_s": alone, "both_wall_s": both_wall, "both_each_s": both_each, "both_over_sum_of_alone": ratio,
            "verdict": verdict}


def write_result(result: dict[str, Any], server: dict[str, Any], root: Path = ROOT) -> Path:
    doc = {"diagnostic": True, "check": "ollama concurrency of the memory and consistency specialists", "server": server,
           "written_at": datetime.now(timezone.utc).isoformat(), **result}
    path = root / "results/local" / f"ollama-concurrency-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def _get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> int:
    from contextlens.agents import consistency_agent, memory_agent
    from contextlens.agents.single import CONTEXT_TOKENS, _user_message, fit_turns
    from contextlens.agents.specialist import _Step
    from contextlens.api import production
    from contextlens.experiments.e7_realtime import _load_dev
    from contextlens.llm.client import OllamaClient
    from contextlens.llm.client import unload_ollama

    sc = _load_dev(ROOT)[0][0]
    user = _user_message(fit_turns(sc.conversation.turns[: sc.request.as_of_seq + 1], CONTEXT_TOKENS), sc.request)
    llm = OllamaClient(production.MODEL, production.OLLAMA_URL, keep_alive=production.KEEP_ALIVE)
    systems = {"memory": memory_agent.system(production.PROMPT_VERSION),
               "consistency": consistency_agent.system(production.PROMPT_VERSION, True)}

    def ask(name: str) -> Callable[[], Any]:
        msgs = [{"role": "system", "content": systems[name]}, {"role": "user", "content": user}]
        return lambda: llm.complete(msgs, schema=_Step, max_tokens=700, timeout_s=60.0)

    calls = {n: ask(n) for n in systems}
    try:
        llm.load()
        for fn in calls.values():  # one untimed call each, so nothing is timed cold
            fn()
        result = measure(calls)
        ps = _get(f"{production.OLLAMA_URL}/api/ps").get("models", [])
        server = {"version": _get(f"{production.OLLAMA_URL}/api/version").get("version"),
                  "loaded": [{k: m.get(k) for k in ("name", "size", "size_vram", "context_length")} for m in ps],
                  "OLLAMA_NUM_PARALLEL_in_this_process": os.environ.get("OLLAMA_NUM_PARALLEL"),
                  "question": sc.id}
    finally:
        unload_ollama(production.MODEL, production.OLLAMA_URL)
    path = write_result(result, server)
    print(path.as_posix())
    print(result["verdict"], f"(both/sum of alone = {result['both_over_sum_of_alone']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
