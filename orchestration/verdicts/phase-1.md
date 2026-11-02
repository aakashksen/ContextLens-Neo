VERDICT: DEFECTS

Third pass over 949e5d1 (code) and 21fc934 (docs). The design (one run record holds many systems) is taken as given. Method: a temp git repo with the real dev (148) and test scenarios and a copy of `src/`, driven through `python -m contextlens.eval.check_run` and `python -m contextlens.eval.claims check`. Probes are in `<local path>` and `probe2.py`. Nothing was written to the real `results/runs/`. The suite passes (165 tests), ruff is clean, and the tree is clean.

## Second-review defects: retested, fixed
- A1 two systems in one record: `check_run E3 --split dev` returns ok, with a different label on each system.
- A2 and A4 test records: a second test record under another name is flagged. The count is across experiments, and `E8x` failed `E8 --split test`.
- A3 earlier bad record: a record with `macro_f1=0.99` followed by a good one still fails. `check_run` checks every record.
- A5 comparison and interval numbers: a comparison with the direction flipped (team minus single saved with single minus team values) fails with "does not recompute". An interval on a system not in the record, a comparison naming a system not in the record, and a comparison of a system with itself all fail.
- B future-turn ranking (`mrr=0.9`), mixed card and card-less rows, and `schema_valid_rate` with raw_output on one row only all fail now. A grounded_rate record without `config.judge` fails. A claim on grounded_rate without `config.judge_review` fails.
- New-hole probes that held: a fake model name nested under a system is caught; a claim with no system name fails when the record has several; a ghost system fails; the wrong system's value fails; a claim on a comparison that is not in the record fails; a supersede that names a record of another experiment is rejected; a superseded bad record cannot be cited in a claim; a later bad record that supersedes the good one is still checked.
- No test was removed or skipped. The test_records.py and test_recompute.py edits are the mechanical move to per-system dicts. The one semantic edit gives all four rows a raw_output, because the new rule drops a metric that is missing on some rows. Every rule in T1.5's list still has a failing case.
- The amended lines in PLAN.md and TESTING.md match the code: seed 13, 10,000 resamples, the supersedes reason, judge and review fields. No thresholds changed. The tasks.json changes are re-verification stamps only.

## Defect 1 (blocking E4, and every comparison on a non-per-scenario metric): intervals and comparisons are not recomputable for the metrics the experiments report
File: `src/contextlens/eval/series.py` lines 15-18 (`PER_SCENARIO`, `BINARY`) and `comparison()` line ~131; `check_run.py` `_comparison_problems` and the interval branch of `check_record`.
Reproduce (in the temp repo):
```
write_run("E4","dev",...,{"single":pa},intervals={"single":{"macro_f1":(0.0,0.1)}})
-> FAIL ... system single: interval macro_f1 cannot be recomputed from predictions
```
`series.is_per_scenario(m)` is False for macro_f1, false_alert_rate, grounded_rate, unsupported_claim_rate, abstain_when_insufficient, answer_when_answerable, speaker_accuracy, latency_p50_ms and latency_p95_ms. Only `label_accuracy`, `mrr`, `recall@k`, `tokens_per_card`, `cost_usd_per_card` and `schema_valid_rate` support a paired comparison. Only those, plus the judged metrics and speaker_accuracy, support an interval.
Impact:
- `docs/TESTING.md` "Comparing systems" requires a 95% interval and a McNemar test. E4's stated headline is macro-F1 (early stopping on dev macro-F1, compared to TF-IDF, zero-shot and prompted). A record cannot hold an interval or comparison on macro-F1. The only workaround is `label_accuracy`, a metric the docs do not name and `recompute_metrics` does not emit (it exists only as a series). So a "trained model beats baseline" claim on macro-F1 has no verifiable interval.
- E5 and E6 compare grounded rate, unsupported claims and abstention against a baseline. Intervals work for grounded_rate and unsupported_claim_rate, but a comparison of them fails, as does one on abstention accuracy. "No measurable difference" cannot be shown in a checked form.
- No plan task owns the fix. T5.2, T6.2 and T7.1 list files that exclude `series.py` and `check_run.py`.
- Fix: say in the plan which task adds macro-F1 (and the other non-per-scenario metrics) to `series.py`. A conversation-level bootstrap of macro-F1 would need a resampling of rows rather than a per-unit mean.

## Defect 2 (blocking E2's cloud comparison, and likely E5 and E7): a record cannot cover a subset of the split
File: `src/contextlens/eval/check_run.py` lines ~178 and ~211 (`n` must equal the split's count; ids must match the split exactly once).
Reproduce: write a record with `dev[:100]` rows, as the T3.3 cloud run asks.
```
FAIL E2-...json: n=100 differs from the split's 148 cases
FAIL E2-...json: system c: prediction ids do not match the split's scenario ids exactly once each
```
Impact:
- T3.3 says "run the best strategy on ClaudeCodeClient over at most 100 ... cases". That record can never pass `check_run E2 --min-n 100 --split dev`. Running all 148 breaks the 100-call cap in the plan and the Claude call cap in `docs/DESIGN.md`.
- T6.2 uses `--min-n 50` and T8.5 uses `--min-n 30`. Both suggest subsets, and both will hit the same failure. The fix would be a record field naming the subset ids (with a manifest or deterministic rule), or a plan statement that these runs are full-split.
- Every system in one record must also cover the same full split. A system that is cheaper on a subset (cloud versus local) cannot sit beside the others.

## Defect 3 (medium): `claims check` does not apply the test-once rule
File: `src/contextlens/eval/claims.py` `check_claims` (calls `check_record` only; `_test_once_problems` is only called from `check_run.check`).
Reproduce: write two test-split records (E8, no `override_note`) and a claim on the second.
```
python -m contextlens.eval.claims check      -> claims ok
python -m contextlens.eval.check_run E8 --split test -> FAIL E8-...: more than one test record and no override_note
```
Impact: a number from a repeated test run reaches `results/claims.json` (and so the README and resume bullets) with `claims check` green. The per-record check also does not look at sibling records, so a claim on a still-valid record passes while a bad sibling stands. Only the test-once case matters here, because it is the protected rule in CLAUDE.md.

## Defect 4 (low, plan gap): calibration and WER/DER have nowhere to be recomputed
`recompute_metrics` emits no ece, wer or der. E4 lists calibration in `docs/TESTING.md` ("for models that output a probability"). T8.5 (E7) asks for word error rate and diarization error. `check_run` rejects any saved metric it cannot recompute ("cannot be recomputed from predictions"), so a record with `ece`, `wer` or `der` fails. `docs/PLAN.md` has no task that adds them: grep for ECE, calibration, WER and DER in PLAN.md and tasks.json finds only the T8.5 prose. T5.2's and T8.5's Files lists exclude `metrics.py`, `check_run.py` and `series.py`. They also need rows to carry the model probability and the ASR/diarization output, and `Prediction` has no field for either. It is not clear which task must add them. Assign them in the plan before T5.2 and T8.5 start.

## Checked and not a defect
- Only T9.1 names the test split (`--split test` at tasks.json:537), so the all-experiment test-once count breaks no other task. The count also includes superseded test records, as documented.
- `supersedes` can remove an earlier record from checking, which is the stated design. The reason is only required to be non-empty, so a weak reason like "x" is accepted. That is a limit of the design, not a bug. The note is printed by the CLI and the file stays.
- `FAKE` matches only "fake" or "stub" (a model named "mock-model" passes). That matches the plan text.
- `judge_review.agreement_rate` of 0.0 with `sample_size` 1 passes `claims check`. The docs say code checks only that these are recorded.
- `tests/test_systems.py` and `tests/test_row_checks.py` are not in T1.5's Files line. Minor.
