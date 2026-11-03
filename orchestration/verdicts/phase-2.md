VERDICT: DEFECTS

Phase 2 review over HEAD 59d9b05 (tree clean). Method: stub embedder, `FakeLLM`, probes in the scratchpad (`...\scratchpad\p2\probe.py`, `leak.py`, `mem.py`). No GPU, Ollama, E1 rerun, or write to `results/runs/`. No QMSum text was copied anywhere.

Checked and fine (no defect):
- `as_of_seq` holds end to end for what exists: `Retriever.search` filters through `store.turns(..., as_of_seq)` in all three modes; `MemoryStore.lookup` filters on `as_of_seq` and computes "superseded" only from items visible at that seq (probe: seq 0 gives m0, seq 1 gives m1, include_superseded gives both). E1 windows take the seq of their last turn, and last-turn queries search only up to `as_of_seq - 1`.
- `Retriever`, `ScoredTurn`, and the `Turn` copy with `speaker="unknown"` line up with what T3.1 (`Tools`, `guard_card`) expects. The Postgres `SELECT` column order matches `Turn.model_fields`.
- Reranker fallback is reported and E1 fails on it. No skipped or weakened tests, apart from the one in defect 3.
- Honesty: README, `docs/*.md`, `docs/RESUME-TARGETS.md`, and `results/claims.json` (empty) say nothing about E1 results. No doc claims more than the records show.

## Defects

### 1. `MemoryStore` has no way to list a conversation's items, so `MemoryExtractor.update(new_turns, existing)` cannot be fed, and item ids then collide
- Files: `src/contextlens/memory/extract.py`, `MemoryStore` (only `add` and `lookup`), and id assignment `f"{conv}-m{len(existing) + len(result)}"`.
- Repro (`...\scratchpad\p2\mem.py`): `dir(MemoryStore)` shows only `add` and `lookup`. Two `update` calls with `existing=[]` (all a caller has without private access to `_items`) both return id `c-m0`. A later correction with `supersedes="c-m0"` then hides both items, because `lookup` builds `replaced` from ids.
- Impact: T3.2's "retrieval + memory" strategy and T4.1's memory agent have to either reach into `_items` or pass a partial list. Either way, ids collide or supersession chains break silently. The wrong "current value" would flow into E2 and E8 grounded and contradiction numbers.
- First task that needs the fix: T3.1 (`Tools.lookup_memory`, which needs a conversation-wide read, and a unique-id scheme). It must be settled before T3.2.
- False number with no later owner: plausible. No later task checks memory id uniqueness, and a wrong-but-valid card would still pass the guard. Memory-strategy numbers in E2 could be affected.

### 2. `unescape(escape(x)) != x`, so `RollingSummary.text` is not faithful
- File: `src/contextlens/llm/untrusted.py`, `unescape` (fixed-point loop); used in `src/contextlens/memory/summary.py`.
- Repro: `unescape(escape("use &lt;b&gt; tags & <b>"))` returns `'use <b> tags & <b>'`. The literal `&lt;b&gt;` the speaker said is turned into `<b>`.
- Impact: low. The summary is a helper and never evidence, and the stored summary is re-escaped on use. Any text that talks about HTML or XML entities is distorted. T3.1's `render_transcript_block` has to reuse `escape` only and must not call `unescape` on anything that could be quoted as evidence.
- First task: T3.1. A false number is unlikely; the effect is bounded to summary wording.

### 3. `tests/test_no_raw_text.py` is probabilistic and can miss copied QMSum/DECODE text
- File: `tests/test_no_raw_text.py:14-15, 46-50`. It uses an 8-word window, keeps one window in four by `hash()`, and Python's string hash is randomized per process.
- Repro (`...\scratchpad\p2\leak.py`): a 14-word passage copied into a tracked file, with `PYTHONHASHSEED=1..12`. It is detected in 10 runs and missed in 2. A shorter quote (8 to 10 words, only 1 to 3 windows) is missed far more often. The plan, T2.5 item 4, says the test "fails if any tracked file holds text" from the raw sources.
- Impact: the guard for the "QMSum and DECODE text never committed" rule can pass while a quote is committed, and its result varies between runs. This is a licensing and data issue, not a number.
- First task that needs the fix: T3.3 (the next task that writes committed output from these sources). Use every window, or a stable hash such as `hashlib`. Cost is small.
- False number: no.

### 4. `configs/retrieval.json` carries no provisional marker for T3.2 to inherit
- File: `configs/retrieval.json` (written by `_config` in `src/contextlens/experiments/e1_retrieval.py`).
- Evidence: it has `winner: "hybrid_t1"` and `scores`, but no `labels_reviewed` (both E1 records have `false`), no flag that the scenarios source is saturated (`dense_t1`, `hybrid_t1`, `dense_w3`, `hybrid_w3` all have recall@5 = 1.0), and no confidence intervals. Only `no_measurable_difference_from` hints at it. On the only source that separates systems, QMSum, `hybrid_t1` 0.768 vs `dense_t1` 0.732 and vs `hybrid_rerank_t1` 0.754 have intervals including 0, so the choice of hybrid is a rule-based pick, not a demonstrated lead. The "mean over sources" rule gives the saturated source half the weight.
- Impact: T3.2 and T3.3 will read `winner`, `mode`, and `rerank: false` as established. A later E2 or README sentence such as "hybrid retrieval chosen by E1" would overstate. Nothing in the repo says this today, so this is a gap and not a false statement.
- First task: T3.2 (it consumes the file). Copy `labels_reviewed` and a `provisional` or `saturated_sources` field into the config, or have the record's `labels_reviewed` re-checked when E2 cites it.
- False number: no number is currently wrong. The risk is a later claim, and `claims.py` would catch only a number, not the word "best".

## Decision on the 6-word overlap note
Not a real concern. The note's overlap is 6-word generic phrasing (a boilerplate "I don't have any personal ..." refusal style) in files we wrote ourselves (`datasets/scenarios/dev.jsonl`, `docs/DESIGN.md`). Such phrases are common in English and in any meeting transcript. The repo's copied-text threshold is 8 words, and 6 is below the length at which overlap indicates copying. Neither the scenario generation nor E1 trains on QMSum, so there is no contamination path. I found nothing to fix there. The real weakness in this area is defect 3: the detector itself is not deterministic.

## Other observations (not defects)
- `fit_context` returns more than `max_tokens` when evidence alone exceeds it. That matches "never drop evidence", but T3.2 should check the length and not assume the budget was met.
- `query_for` (the rule for building a query from a press) lives only in the E1 module. T3.2's `RetrievalAssistant` has to reuse or restate it, or E2 will not match E1's setup.
