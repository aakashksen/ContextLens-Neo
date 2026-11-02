VERDICT: DEFECTS

Checked: docs/PLAN.md T1.1-T1.5, the T1.* verdicts, and an end-to-end run in a temp git repo (script at /tmp/e2e.py, nothing written to the real results/runs/). The run loaded the real 148 dev scenarios, replayed them with a simple fake assistant, computed metrics, wrote a record with write_run, and ran check_run on it. With only recomputable metrics, check_run passes (`[]` with min_n=100, split=dev). Splits, hashes, dirty check and the ids-exactly-once check behave as the verdicts say. I found no leakage: replay truncates at as_of_seq, and grounded_rate rejects future turns. I found no skipped tests.

## Defect 1 (blocking for later phases): check_run rejects any saved metric it cannot recompute, so most metrics the plan needs for E1-E8 cannot be recorded
- File: src/contextlens/eval/check_run.py, `recompute_metrics` (lines ~28-48) and the loop in `check_record` that emits "metric {name} cannot be recomputed from predictions".
- Reproduce: in a temp repo, use the e2e script. Write a run whose metrics hold the recomputable ones plus `grounded_rate=1.0`, which is a correct value from `metrics.grounded_rate`. `check("E9", min_n=100, split="dev")` returns `['...: metric grounded_rate cannot be recomputed from predictions']`. Without the extra key it returns `[]`.
- Recomputable set: macro_f1, false_alert_rate, abstain_when_insufficient, answer_when_answerable, latency_p50_ms, latency_p95_ms. Not recomputable: recall@k, mrr, grounded_rate, speaker_accuracy, tokens, cost, schema-valid rate and unsupported-claim rate, which PLAN names for T2.5, T3.3, T6.2 and T8.5.
- Impact: every experiment's acceptance command (`check_run E1/E2/E5/E7 ...`) will fail if its record holds those metrics. The implementer's options are to drop them from the record, which hides the headline metrics, or to widen the recompute function without review. Claims can only cite metrics present in a record, so the headline numbers cannot reach `results/claims.json` honestly. E1 retrieval also does not fit at all: its output is ranked turns, not `Prediction` cards, and `check_record` requires Prediction rows. Fix: either extend `recompute_metrics` (grounded_rate, speaker_accuracy and token/cost sums from cards; retrieval ranks need a predictions format for E1), or have check_run recompute what it can and explicitly list unverifiable metrics as unverified. Do this before T2.5.

## Defect 2 (medium): zip-based metrics silently truncate on length mismatch and give a wrong, high score
- File: src/contextlens/eval/metrics.py, `macro_f1`, `false_alert_rate`, `abstention_accuracy`, `speaker_accuracy` (lines ~27-75).
- Reproduce: `speaker_accuracy(["a","b","c"], ["a"])` returns 1.0, since the extra predictions are dropped. `macro_f1(["update","update"], ["update"])` returns 0.25 and ignores the missing prediction. `speaker_accuracy(["a"], ["a","b"])` returns 0.5.
- Impact: a misaligned list from a future experiment gives a plausible number with no error. The number would be saved in a record, and check_run would pass it if it uses the same function. Raise ValueError on unequal lengths, as `paired_bootstrap_diff` already does.

## Defect 3 (low): empty input returns 0.0, which looks like a real score
- Same file: `mrr([])`, `grounded_rate` with no findings, `speaker_accuracy([], [])`, and both sides of `abstention_accuracy` with no cases all return 0.0. `bootstrap_ci([], [])` raises ZeroDivisionError.
- Impact: an experiment that produces no findings, for example a model that abstains everywhere, reports grounded_rate=0.0, which is indistinguishable from "all findings ungrounded". It could end up as a claim. Return None or NaN, or raise, and document it.

## Defect 4 (low, becomes real in E7): Prediction.stage_latency_ms holds only "total"; the error card is always mode "recall"
- File: src/contextlens/replay/runner.py lines ~46-52. The e2e run shows `{'total': 0}` for every prediction. `recompute_metrics` reads only "total".
- Impact: T8.5 (E7) must report p50/p95 by stage, and PLAN T4.1 requires per-agent timing. Replay gives no stage hook (Context has no timer), so assistants cannot fill stages in through the documented interface. The abstained error card uses `mode="recall"` whatever was asked, so error cards for clarify/respond cases are mislabelled (a category breakdown by mode would be wrong). Neither blocks Phase 1 acceptance. Fix by letting the Assistant or Context report stage timings and by passing the intended mode through (or by recording the mode as unknown).

## Not defects (checked)
- The `Fake|stub` regex also matches model names that merely contain "fake" or "stub". This is conservative and fails closed.
- Manifest trust, and a top-level-list claims.json raising AttributeError: both fail closed (nonzero exit).
- Dataset hash ignores timings: documented.
- README and docs claim no numbers. `results/claims.json` is `{"claims": []}` and `results/runs/` is empty, so no claim exceeds the runs.
