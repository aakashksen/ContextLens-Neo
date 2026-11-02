VERDICT: DEFECTS

Second pass. I checked the four fix commits (86f8859, 1b693ad, 72d23e7, 76f2616) against the first review's defects. I also reran the full path on the real 148 dev and test scenarios in a temp git repo. The probe script is at `<local path>`. It wrote nothing to the real `results/runs/`. The suite passes (105 tests), ruff is clean, and the tree is clean.

## First-review defects: retested, fixed
- D2 (length mismatch): `speaker_accuracy(["a","b","c"],["a"])` and `macro_f1(["update","update"],["update"])` now raise ValueError.
- D3 (empty input): `mrr([])`, `speaker_accuracy([],[])`, `grounded_rate` with no findings, `abstention_accuracy([],[])` and `macro_f1([],[])` return None. `bootstrap_ci([],[])` raises ValueError. `claims.py` rejects a claim on a None metric. Saved None only matches a fresh None, and a saved 0.0 for an undefined metric is rejected.
- D4 (stages/mode): `Context.stage_ms` flows into `stage_latency_ms`. Error cards carry `errored=True`. The card's mode is still the placeholder "recall"; this is now flagged rather than corrected. I count that as acceptable.
- D1 (unrecomputable metrics): grounded_rate, unsupported_claim_rate, speaker_accuracy, tokens/cost per card, schema_valid_rate, recall@k, mrr, per-stage latency, and `[category=]` / `[origin=]` breakdowns now recompute. End-to-end on the real dev scenarios with a fake assistant, `check_run` returns `[]`. A made-up suffix (`macro_f1[category=bogus]`) and a wrong-k value (`recall@3` set to 0.5) are both rejected. Old prediction rows without the new fields still load.

## Defect A (blocking for E3 and E8, needs a design choice): check_run cannot verify a system comparison
Files: `src/contextlens/eval/check_run.py` (`check`, lines ~168-180; the ids-exactly-once check in `check_record`), `src/contextlens/eval/records.py` (`write_run`).

The record format holds one predictions file, and `check_record` requires each scenario id to appear exactly once. `check` verifies only the latest record. `intervals` is never read.

Reproduced in the temp repo:
1. Two systems in one record. Passing `preds + preds` to `write_run` fails with `n=296 differs from the split's 148 cases` and `prediction ids do not match ... exactly once each`. A paired record, which T4.2 asks for ("fixture writes a valid paired record"), cannot hold both systems' predictions.
2. One record per system on the test split (E8, T9.1, seven systems). Writing two E8 test records gives `more than one test record and no override_note in config`. Seven systems would need six override notes. That defeats the once-only test rule, and it would hide a real second test run.
3. One record per system on dev. I wrote an E3 record with `macro_f1=0.99` (wrong), then a correct E3 record. `check E3 --split dev` returns `[]`. Only `recs[-1]` is checked, so earlier records are never verified.
4. Distinct experiment names do not escape the glob. `records_for` matches `E8-*Z.json`, so a record named `E8-recent-<ts>Z.json` joins the same group.
5. Unverified comparison numbers. I wrote `intervals={"macro_f1": (0.99, 1.0), "diff_vs_baseline": (0.5, 0.6)}` with real metrics, and `check E4` returned `[]`. Paired bootstrap differences, McNemar p-values and CIs are plain assertions. TESTING.md requires them to compare systems ("Comparing systems"), and PLAN T1.2 says a saved metric must recompute. `claims.py` compares a claim against `metrics` only. An improvement claim from E3, E4, E5 or E8 therefore has no verified basis.

Impact: the acceptance commands `check_run E3 ... --split dev` and `check_run E8 ... --split test` either fail or pass without checking the comparison. This surfaces at T4.2 at the latest.

Options (a design choice for the person or orchestrator, not made here):
- (1) One record holds a map of system name to predictions file. `check_run` verifies every system (ids exactly once, metrics recompute per system) and recomputes `intervals`. Intervals can be recomputed deterministically: seed 13, 10,000 resamples, with `paired_bootstrap_diff` and `mcnemar_exact` taking correctness arrays derived from the rows. The once-only test rule then applies per record.
- (2) One record per system with a `system` or `group` field. `check_run` verifies all non-superseded records for the group. The test-once rule counts per system, plus one comparison record whose `intervals` are recomputed from the referenced records' predictions.
- (3) Keep records per system and require that comparison figures come only from a separate verified analysis. This needs a stated rule that `intervals` must not be cited in claims until verified, and check_run still needs to stop reading only `recs[-1]`.
Whichever is chosen, PLAN T1.2 / T4.2 / T9.1 and `docs/TESTING.md` need amending, because the "one predictions file" assumption is baked into them.

## Defect B (medium): rows can carry data from the future or be selectively omitted, and still pass the recompute
Files: `src/contextlens/eval/check_run.py` (`_base_metrics`).

New recompute paths trust fields in the predictions file with no cross-check. Reproduced on real dev:
- Retrieval rows past `as_of_seq`. Rows with `ranked_turn_ids` listing the entire conversation, including turns after `as_of_seq`, give mrr=0.90 and recall@1=0.83 on dev. `check_run` accepts the record (`[]`). Ids not in the conversation (`"bogus"`) are also accepted. E1 retrieval does not go through `replay`, so nothing else enforces the cutoff. `grounded_flags` rejects future turns for cards, but `recall_at_k` and `mrr` do not. `speaker_accuracy` also counts card evidence on future turns, since it only skips ids missing from the conversation.
- Mixed card and card-less rows. Half the rows were card-less (`ranked_turn_ids=[]`). `check_run` passed `macro_f1` and `answer_when_answerable` computed on only the carded half (macro_f1 0.09 vs 0.086 on all). Dropped or card-less rows silently leave the denominator. `n` still counts them.
- `schema_valid_rate` over rows that have `raw_output`. With raw_output on only 1 of 148 rows, `schema_valid_rate=1.0` is accepted. Rows without raw_output are ignored, so failures can be omitted.
- `claim_supported` all True is accepted with `grounded_rate=1.0`. The judgement is a saved input (the rubric judge plus person review described in TESTING.md), so `check_run` cannot verify it. Nothing ties it to a judge model or a review sample. This is unavoidable, but a headline grounded_rate is only as good as that field. The claim path should require a recorded judge and review before such a claim.

Impact: honest-looking numbers can pass the gate that was meant to catch them. Suggested fix: reject `ranked_turn_ids` entries that are not turns at or before `as_of_seq`, and card evidence turns after `as_of_seq`. Either require all rows (or all rows of a stated kind) to carry cards, `raw_output`, and judgements, or put the row count behind each metric in `metrics`.

## T1.3 Interfaces and scope
- `Context(store, as_of_seq)` and `replay(...)` signatures hold, and `Context.stage_ms` has a default. But `Prediction.card` is now `Card | None`. PLAN T1.3 says `Prediction: scenario_id, card, stage_latency_ms`. Any later code doing `p.card.findings` needs a None guard, and none of it exists yet. `errored`, `ranked_turn_ids`, `claim_supported` and `raw_output` go beyond that interface. They are what the recompute fix needed and are used by tests, so I do not count them as bloat. The plan text should be updated to list them.
- `check_run` now recomputes many metrics by category and origin (`name[category=x]`). Nothing in the plan asks for these breakdowns by name. T4.2 does ask for a category breakdown, so this is defensible.

## Honesty
No README or docs claim numbers, `results/claims.json` has no claims, and `results/runs/` is empty. Nothing exceeds the runs.
