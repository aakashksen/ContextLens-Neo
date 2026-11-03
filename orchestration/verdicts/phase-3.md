VERDICT: DEFECTS

Phase 3 review at build commit 61d06d4. Scope: T3.1 to T3.3, record E2-20261103T155026435232Z and the two comparison records, `results/error-analysis-e2.md`.

What I ran: `tests/test_e2.py tests/test_single_agent.py tests/test_guard.py tests/test_tools.py tests/test_no_raw_text.py` with `PYTHONPATH=src` (172 passed, 115 s). I recomputed counts from the saved prediction rows. I did not run `check_run E2`. The baseline log `orchestration/evidence/cleanup-baseline.raw.txt` held only the commit line when I looked, so I did not use it. No GPU, Ollama, network or Claude calls. I opened no test split and read no QMSum or DECODE text.

What holds up:
- The numbers in the analysis agree with the saved rows for the figures I rechecked (answer rates, health counts, modes per category).
- No test-split use, no leakage found. `build_store` and `build_memory` cut at `as_of_seq`, and the cloud subset is dev-only, synthetic and committed before the cloud run.
- The tried-prompt record is labeled as such. The comparison records keep the winner basis, and the `retrieval_memory` choice is disclosed as coming after dev results.
- No weakened or skipped tests found. `claims.json` is empty, and README and RESUME-TARGETS carry no E2 numbers.

## Defects

### D1. Result-affecting (unquantified): the number rule drops statements that merely mention a turn id or a count word, and the analysis never decides about it
- Files: `src/contextlens/agents/single.py` lines 165-196 (`_runs`, `_numbers_supported`, `verify_card`); `results/error-analysis-e2.md` lines 46-57.
- Reproduce, with `PYTHONPATH=src`:
  `from contextlens.agents.single import _runs; _runs("Abel changed his answer in turn t3 (syn-con-04:t1)")`
  returns `{('3',), ('1',), ('04',)}`, and `_runs("in the first of two turns")` returns `{('two',)}`. A true statement that names a turn or says "two" is dropped, unless the quote happens to contain that same token.
- Evidence in the records: `recent_cloud` has 16 of 100 cards carrying "finding(s) dropped: a number in it is not in the cited quotes". 12 of them ended abstained on answerable cases (the analysis lists the ids). The dropped text is not saved, so I cannot say how many drops were correct. Several of these cards also have `missing` notes about `start_ms` being 0, so the model may be writing ids and timestamps into statements.
- Impact:
  - The cloud answer rate (0.831) and the local-vs-cloud comparison are understated by an unknown amount.
  - PLAN.md line 187 says "the data decides whether to revisit T3.2's rule". The analysis counts drops and "unsupported though it passed" cases, but never measures the false-drop side and never states a decision. That decision is still open at phase end.
- Fix needed: read the 12 cloud cases (synthetic text, so allowed) and save dropped statement text for non-restricted sources. Then state keep, change or defer in the analysis.

### D2. Lesser: abstentions from the model and from the guard are merged, so "answer rate" can't be read as model behavior
- Files: `src/contextlens/agents/single.py` `verify_card` lines 255-265; `results/runs/E2-20261103T124545651045Z.*.predictions.jsonl`; `results/error-analysis-e2.md`.
- Reproduce: count abstained rows whose `missing` holds "no finding passed the checks" but no "number in it" note. `recent`: 37 of 95 abstentions. `retrieval_memory`: 18 of 26. `raw_output` is null in the rows.
- Impact: these are cases where the model produced findings and the citation guard (quote not a substring of the cited turn, or an unknown id) dropped all of them. This is about 25% of `recent` cases and a large share of the answer-rate gap between strategies. The analysis lists the number-rule side as "not measured" but says nothing about guard drops. A quote-matching problem, for example a paraphrased or truncated quote, is indistinguishable from a model that chose to abstain. The "winner" and the answer rates rest on that mix.

### D3. Lesser: the tried-prompt record cannot be told apart from the baseline by its prompt label
- Files: `results/runs/E2-20261103T155026435232Z.json` (`prompt_versions` is `{"recall": "v1"}`); commit c9fe70e changed the prompt inline in `single.py`, not in a registry version.
- Impact: two different prompt texts share the label v1. Only `config.kind`, `prompt_fix_commit` and the analysis separate them. The code is now reverted, and `git_commit` fda76e6 is the only way to rebuild it. This is honest in the prose but weak as a record. The run record's reproducibility rule is met only through git history.

### D4. Lesser: a memory-build failure silently turns `retrieval_memory` into plain retrieval for that case
- File: `src/contextlens/experiments/e2_context.py` lines 404-415 (`build_memory`: `except Exception: return MemoryStore(), None`).
- Evidence: the record has `memory_build.failed = 2`, and the failing ids are not saved.
- Impact: 2 of 148 cases of the strategy the design now carries forward were measured as a different strategy. Small, but unlabeled per case.

### D5. Lesser: the public-data slice is tiny and the analysis does not say so
- Evidence: the metrics `[origin=dataset]` in the E2 local record show answer rates in tenths (0.5, 0.7), which implies about 10 answerable public cases, and the dataset-origin `abstain_when_insufficient` is `None`.
- Impact: the plan asks for a split by public vs synthetic. The split exists, but the analysis never states n or that no public case tests abstention, so the public-vs-synthetic rows can't support any comparison.

### D6. Lesser: the injection requirement is unmet at phase end and its carry-forward is thin
- Review focus 4 in PLAN.md ("injected speech doesn't change the card or its evidence") is unmet on dev for the system the coordinator chose. The analysis (and my re-read of the verdict) shows real adopted injected claims: `retrieval_memory` syn-inj-02, syn-inj-10 and syn-inj-14 in findings, and syn-inj-02 also in a suggestion on the tried prompt.
- The carry to T4.2 (instruction guard, tested on syn-inj-02 plus look-alikes) covers the guard. It is placed after T4.1, which passed using only a `FakeLLM` injection test, so the team is built before the guard exists. This is adequate as a plan if the T4.2 guard goes into the shared `guard_card`. The plan should say so, or T4.1's team will bypass it.
- The suggestion step was never exercised by a real model (respond on 0 of 148 `retrieval_memory` cards, 1 of 100 cloud cards). So "no injected claim reaches a suggestion" is unmeasured. The analysis says so, and T4.2 and T7.1 carry it. The respond-in-code decision at PLAN.md line 220 is the first point where it can be measured.

## Not defects (checked)
- No raw QMSum or DECODE text in committed E2 files; only synthetic scenario text is quoted.
- The claims counter (203 calls, 101 used by E2) is consistent with the `.carried` file (102).
- `interview-notes.md` does not exist yet, and the analysis says so.
