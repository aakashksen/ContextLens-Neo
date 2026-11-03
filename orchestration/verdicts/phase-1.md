VERDICT: DEFECTS

Fourth pass over HEAD cc1f424 (tree clean). The earlier decisions are taken as given: one record holds many systems, and DECODE and QMSum are approved for local, non-commercial evaluation with their data never committed.

Method: a temp git repo with the real dev (148) and test scenarios and a copy of `src/`, driven through `python -m contextlens.eval.check_run` and `python -m contextlens.eval.claims check`. The probe is `<local path>`, and `p3.py` for the E1 case. Nothing was written to the real `results/runs/`.

## 1. The four previous defects, retested through the CLIs
- Old defect 1 (macro-F1 and other metrics had no interval or comparison): fixed. A two-system E4 record with a macro_f1 interval, false_alert_rate interval, and comparisons on macro_f1, false_alert_rate, abstain_when_insufficient, answer_when_answerable, latency_p50_ms and `macro_f1[category=contradiction]` returns `E4 ok`. It took about 30 s.
- Old defect 2 (a record could not cover a subset): fixed. A 100-id subset of dev, with `datasets/splits/subsets/e2-cloud.json` committed before the run, returns `E2 ok` with `--min-n 100 --split dev`.
- Old defect 3 (`claims check` ignored test-once): fixed. With two test records and a claim on the second, `claims check` gives `FAIL claim c: run: E8-...: more than one test record and no override_note in config`, and `check_run E8 --split test` fails too.
- Old defect 4 (no ECE, WER or DER): the plan now assigns them. PLAN.md T5.2 and T8.5 have "From phase review" lines that add the metric, the check and a row field. A saved `ece` is still rejected today with "cannot be recomputed from predictions", which is expected until T5.2.
- Retrieval-only E1-style record (recall@k, mrr, intervals, a McNemar comparison on recall@3) passes with `--min-n 100`, but only after fixing defect 4 below.

## 2. Can each Experiment task pass `check_run`?
- E2, E3, E4, E5, E6 pass in form. Every metric the plan asks for is recomputable, and intervals and comparisons work for all of them. Two cautions follow. T3.3's cloud run needs a committed subset manifest, and T3.3's text does not say so (only TESTING.md does). The E5 and E6 judged metrics need `config.judge`, and claims need `config.judge_review`.
- E1 passes only on committed scenarios (defects 1 and 4).
- E7 is blocked as written (defect 5).
- E8 can only cover what is in `datasets/scenarios/test.jsonl` (defect 1).

## Defects

### Defect 1 (blocking T2.5 as specified): a record on QMSum or DECODE text cannot pass `check_run`, and the plan has no mechanism for it
Files: `src/contextlens/eval/check_run.py` lines 197-224 and `records.py` `split_hash`. `check_record` reads scenarios only from the committed `datasets/scenarios/<split>.jsonl`, pinned by the committed `datasets/splits/scenarios.json` hash. Prediction ids must equal that file's ids exactly once each, and gold labels, turns and `as_of_seq` all come from it. A named subset only narrows ids inside that file.
Reproduce: in the temp repo, put a scenario `qmsum-val-0-q0` in a local file `datasets/scenarios/qmsum-dev.jsonl`, add a row for it to a dev record, then run `check_run E1 --split dev`. Result: `FAIL ... n=149 differs from the split's 148 cases | system a: prediction ids do not match the split's scenario ids exactly once each`. The extra file is never read. Today's dev.jsonl and test.jsonl hold only synthetic (138 and 136) and LongMemEval (10 and 10) scenarios.
So an experiment on QMSum or DECODE can reach an accepted record only by putting their text into the committed dev.jsonl or test.jsonl. PLAN.md T2.5 ("using QMSum and our scenarios"), T6.1 (QMSum cards), T9.1 ("report separately for each public benchmark") and the E5 and E6 cases depend on this. Also, any edit to dev.jsonl or test.jsonl changes the hash. Every earlier dev record then fails "dataset_hash does not match the split manifest", and the H2 label review covers the old test set.
What is open: nothing says that `check_run` should accept scenarios from an uncommitted overlay (ids, counts and a hash in git, text under `datasets/raw/`), or that those experiments report on committed scenarios only.
First task needing the fix: T2.5 (E1). Next are T6.1, T6.2 and T9.1.
False number risk: not through `check_run`, which rejects the rows. The risk is mislabelling. An E1 implementer who cannot score QMSum can run on the 148 committed scenarios while the plan, config or a claim says "QMSum". `check_run` cannot check a dataset name in `config`, and the origin breakdown would show only longmemeval and synthetic. T9.3 owns the report but not this check. The other way out is committing the text, which is defect 2.

### Defect 2 (the protected rule on DECODE and QMSum text has no guard): nothing stops their text being committed, and several planned outputs have unspecified paths
Evidence:
- `.gitignore` ignores `datasets/raw/` only. `git check-ignore` shows `datasets/scenarios/qmsum-dev.jsonl` and `results/runs/x.json` are not ignored. The only hook in `.git/hooks` is `commit-msg`, which strips attribution lines. No test, taskctl step or tool scans committed files for DECODE or QMSum text.
- `tests/test_scenarios.py` line 52 does fail if a dataset-origin scenario's conversation is not in the longmemeval manifest, so a QMSum scenario in dev.jsonl breaks that test. It is still a lone accidental barrier, and a task that needs QMSum scenarios would loosen it (the weakened-check pattern).
- `results/runs/*.predictions.jsonl` is committed, and each row holds card `evidence.quote` and `raw_output`. Prediction rows in an E1, E5 or E6 run on QMSum would hold its text.
- Output locations are not named in PLAN.md for T6.1's training examples (`cards/dataset.py` has no output path), for `results/error-analysis-e4.md` ("50 mistakes", which could come from DECODE), for T5.1's `build_pairs`, and for `configs/`.
Impact: the person's condition ("never committed") can be broken by an ordinary commit or by the end-of-task commit, with every gate green. The text would also be in history.
First task: T2.5, then T5.1, T5.2 and T6.1. Suggested fix: add a test or taskctl check that fails on any tracked file containing text from `datasets/raw/qmsum` or `datasets/raw/decode`, and add the output paths to the plan and `.gitignore`.
False number risk: none. It is a leak risk.

### Defect 3 (plan gap; first needed at T2.5 for QMSum and T5.1 for DECODE): no task downloads DECODE or QMSum, confirms their real format, or makes their split manifests
Evidence:
- T1.1 is done. It says to download only approved datasets, and DECODE and QMSum were not approved then. `datasets/raw/` holds only `ami/` and `longmemeval/`.
- `datasets/splits/` holds `ami.json`, `longmemeval.json` and `scenarios.json`. `src/contextlens/data/splits.py` `__main__` writes only longmemeval and ami. `taskctl status` shows 8 done, so T1.1 will not run again.
- The DECODE loader (`loaders.py` `_decode_records`, `load_pairs`) reads `decode/*.jsonl` with fields `id`, `dialogue[{speaker,text}]` and `is_contradiction`. `docs/datasets.md` line 24 says "Format: Couldn't confirm", and only the fixture `tests/fixtures/data/decode/train.jsonl` uses this layout. The real DECODE comes by ParlAI or a raw S3 download. The QMSum loader reads `qmsum/*.jsonl` and treats `relevant_text_span` as turn indices, also checked only on a fixture. `load_gold("decode")` does not exist, and `_gold_qmsum` ignores QMSum's own train, val and test files when `make_splits` re-splits.
- T5.1 "Consumes `Pair`, `load_pairs` from T1.1" and says nothing about downloading, format checks or manifests. T2.5 and T6.1 say nothing either.
Impact: the first task to need them either blocks, or an implementer adapts the loader inside an unrelated task. If the guess about QMSum spans or the DECODE label meaning is wrong, gold evidence or training labels are wrong and the verifier sees only fixtures.
False number risk: a wrong QMSum span mapping could give a wrong recall number that no later task catches, because `check_run` trusts the scenario file. A wrong DECODE mapping affects training data, and E4's number is measured on the committed scenarios, so it stays honest but poor. The risk is real but not demonstrated, because nobody has run the loaders on real data.
Fix: add an owner (a small data task before T2.5, or a clause in T2.5 and T5.1) to download under `datasets/raw/` (git-ignored), print the real schema, adjust the loaders, write the manifests with ids, counts and hashes only, and add a test that the manifests hold no text.

### Defect 4 (low; T2.5 hits it first): a retrieval-only record fails with a judge demand if metrics come straight from `recompute_metrics`
Evidence: T1.5 verifier note 3, reproduced. `recompute_metrics` on rows with no cards emits `grounded_rate: None` and `unsupported_claim_rate: None`. A record that saves them, as my first E1 probe did, fails with `FAIL ... config.judge is missing: a record with grounded_rate or unsupported_claim_rate must name its judge`. `check_run.py` lines 286-288 test whether the name is saved, not whether the value is not None. After dropping None-valued metrics, the same record gives `E1 ok`.
Impact: the first E1 write fails with a message that points to a judge that does not exist. A hasty fix would be a made-up judge in `config`. T2.5's test ("tiny fixture writes a valid record") would show it, so it should be found at once. No false number is possible: the metrics are None and a claim on a None value is refused.
Fix: ignore None-valued saved metrics in the judge test, or say in T2.5 to omit them.

### Defect 5 (blocking T8.5; found by reading and the existing unknown-turn rule): E7's ASR transcripts and AMI questions have no place in the dev scenario set
Evidence: `_row_problems` (`check_run.py` lines 131-145) fails a row that cites a turn id not in the scenario. The earlier "ghost turn fails" probe still holds. T8.5 replays AMI audio through speech and diarization and compares "the same questions as on the clean transcript". ASR turns are segmented differently, so a row on the ASR transcript cites turn ids that are not in the clean scenario. dev.jsonl holds no AMI scenario (counts above), and adding one changes the dev hash (see defect 1). I did not run an ASR case; this follows from the code.
Impact: `check_run E7 --min-n 30 --split dev` cannot pass for the ASR system. The WER and DER row fields that T8.5 adds do not fix this.
First task: T8.5, though the scenario decision belongs earlier. False number risk: none, because `check_run` rejects it. It is a blocked task.

## Checked, not defects
- T1.5 verifier notes 1, 2, 4 and 5. Note 2 (claims on the earliest test record pass while a later un-overridden one exists) follows the plan's wording, and the number cited is the first, legitimate test use. Note 4 (a hand-forged record with a non-matching filename is not seen) needs a forged file, and every writer goes through `write_run`. Notes 1 and 5 are not product defects.
- No test was removed or skipped in the recent `fix(eval)`, `fix(claims)` and `docs:` commits. `results/claims.json` has no claims and no run records are tracked, so no unearned number exists.
- The T3.3 cloud run (at most 100 cases) needs a committed subset manifest before the run. TESTING.md states the rule, T3.3 does not repeat it.
