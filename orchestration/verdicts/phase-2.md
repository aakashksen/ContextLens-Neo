VERDICT: DEFECTS

Phase 2 re-review at HEAD 7c96836 (tree clean). The four earlier defects are fixed. Two new, minor defects are below, and neither can put a false number on record. Method: stub data, `FakeLLM`, and probes in the session scratchpad (`...\scratchpad\mem2.py`, `leak2.py`, `cfg.py`). No GPU, no Ollama, no E1 rerun, no write to the real `results/runs/` (config regeneration ran on a temp copy of the records), and no QMSum text was copied anywhere. `tests/test_memory.py` and `tests/test_e1.py` pass (35 tests). `tests/test_no_raw_text.py` passes with 10 tests, and the real-corpus test ran and was not skipped, because `datasets/raw/qmsum` is present.

## Earlier defects, retested

1. Memory ids: fixed.
   - In the real flow, extract then add, then extract with `existing=store.items(conv, big)`, then add, the ids come out `c1-m1-0` and `c1-m2-0`. The correction's `supersedes` is `c1-m1-0`, and `lookup` returns only "Meet on Wednesday".
   - A third update over the same newest seq gets `c1-m2-1`, with no collision.
   - `MemoryStore.items(conv, as_of_seq)` filters by conversation and seq, includes superseded items, and sorts by seq.
   - `add` is all-or-nothing and raises `ValueError` on a duplicate id.
   - Nothing raises in an ordinary flow. The one way to make it raise: pass `existing` filtered to an older seq (for example `items(conv, 1)`) when the store already holds items with the same newest seq. The two updates then both mint `c1-m2-0`, and `add` raises. That is a misuse that fails loudly, not a silent wrong answer. T3.1 and T3.2 should feed `existing` from the store at the current seq.
2. `unescape`: exact and fixed. `unescape(escape(x)) == x` for the probes, and `unescape("&amp;lt;") == "&lt;"`. In `_from_model` the speaker's literal `&lt;b&gt; tags & <b>` survives. The summary is escaped again by `escape(self.text)` on its only reuse, so no route puts an unescaped delimiter into a prompt. A model echo of a raw `</transcript>` stays in `self.text`, which is only a Python attribute, and is escaped on reuse. T3.1 must keep passing it through `escape` and never interpolate `RollingSummary.text` raw. Defect A below is a remaining fidelity gap.
3. Leak test: deterministic and fixed. The 8 `PYTHONHASHSEED` variants pass, because the hash is now `blake2b` over every window. It scans JSON and JSONL strings recursively, and unparseable JSON lines fall back to the raw line. In my probe a 9-word copy inside a nested JSON string, a `.jsonl` line, an ASCII-escaped JSON string and a plain `.txt` was caught every time. The file list is `git ls-files --cached --others --exclude-standard`. The committed `results/` files, `configs/` and `datasets/scenarios` are included, and `datasets/raw/` is git-ignored. Defect B below is a remaining gap.
4. `configs/retrieval.json`: fixed.
   - The change was purely additive. Every old key has the same value, and `winner` is still `hybrid_t1` with the same scores. The new keys are `labels_reviewed` (false), `not_measurably_better_than_statement`, `saturated_sources` (["scenarios"]) and `saturation_note`.
   - `regenerate_config(tmp_root)` reproduces the committed file byte for byte from the records. It reads only records and changes nothing in them.
   - The statement matches the record intervals. On qmsum, `hybrid_t1` against `dense_t1` is [-0.005, 0.077] and against `hybrid_rerank_t1` is [-0.024, 0.053], both including 0.

`query_for` move: no change to E1's behavior. The body is identical, `e1.query_for is query.query_for` holds, and its only caller is still E1 at `e1_retrieval.py:118`. T3.2 can now import it from `contextlens.retrieval.query`.

## Defects

### A. `_from_model` loses a literal `&amp;...` the speaker said
- File: `src/contextlens/memory/summary.py:11-14`, the regex `&amp;(?:amp;)+` that collapses over-escaping.
- Repro: for a speaker's literal text `AT&amp;T`, the prompt holds `AT&amp;amp;T`. If the model echoes that, `_from_model` returns `AT&T`, not `AT&amp;T`. A literal `&amp;lt;` comes back as `&lt;` (the original has no `<`), and a literal `&amp;amp;` comes back as `&`. Probe: `_from_model(escape(lit)) == lit` is False for those three and True for `&lt;b&gt;` and `</transcript>`.
- Impact: low. Only text that talks about HTML entities is distorted, in a summary that is never evidence. The docstring claims only that a literal `&lt;` survives. The gap is that an echo of an escaped `&amp;amp;` cannot be told from over-escaping. No delimiter is created.
- First task that needs the fix: none required. If T3.1 or T4.1 ever shows the summary to a user, it must not be treated as a verbatim quote.
- False number: no.

### B. The leak detector misses hard-wrapped quotes and quotes in JSON keys
- File: `tests/test_no_raw_text.py`, `_texts` (lines 29-41). Markdown and other text files are scanned one line at a time, and only JSON string values are scanned, not keys.
- Repro (`...\scratchpad\leak2.py`): a 10-word passage from a raw file, wrapped over two 5-word lines in a `.md`, is missed. A quote of 9 words split across two JSON fields is missed (expected, since it is not contiguous), and a quote used as a JSON dict key is missed. Quotes on one line, in JSON/JSONL strings, in `.txt`, and ASCII-escaped JSON are caught.
- Impact: a short quote of up to about 14 words, wrapped at the line width, would pass the guard that backs the "QMSum and DECODE text never committed" rule. It is unlikely, because the repo's docs are mostly one paragraph per line. This is a licensing and data issue, not a number.
- First task that needs the fix: T3.3, the first task that writes committed prose or output derived from these sources. Join the lines of non-JSON files into one word stream, and scan dict keys too.
- False number: no. No later task owns this, but its effect is bounded to the guard.

## Other observations (not defects)
- `fit_context` can return more than `max_tokens` when evidence alone exceeds the budget, so T3.2 should check the length. This carries over from the first review.
- The `update` prompt writes `existing` item ids and `conversation_id` unescaped (`extract.py` `have`). Ids are minted by our own code from dataset conversation ids, so there is no current route for untrusted text. T3.1 should use `escape` if conversation ids ever come from user input.
