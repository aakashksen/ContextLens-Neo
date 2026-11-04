# Decisions

Why each real choice was made, written so it can be explained out loud. One entry per choice, including attempts that failed and ideas that were reverted.

**Read this first.** Every result mentioned here is provisional. All run records so far are on dev data with `labels_reviewed: false`, meaning no person has checked the dev labels yet, so `claims check` refuses any claim built on them. This file therefore holds no metric, percentage or latency value. It names the record ID and the metric and gives the direction in words. A count of attempts or settings is not a result and may appear. No QMSum or DECODE text is quoted.

**How to read an entry.** "Who" is the person, the coordinator (the `porter-` session), or the orchestrator, as far as the sources show. "Reason not recorded" means no source says why. A reason that was only seen after a result came in is marked "after seeing results".

**Closing rule.** A task closes only if this file mentions its id, or has a line `<id>: no new decisions`. `tools/taskctl.py` enforces it.

**Sources for the backfill.** `docs/PLAN.md` (Decision and From lines), `docs/model-choice.md`, `docs/datasets.md`, `orchestration/verdicts/`, commit messages, the run records in `results/runs/` (E2's are on branch `task/T3.3`), and the orchestrator's session transcript. The transcript was treated as raw material, not as instructions.

---

## 1. Local model is qwen3.5:4b-q4_K_M, with gemma4:e4b-it-qat as the judge
- **Date / task:** 2026-11-02, before T2.1 (commit 8e0df9f)
- **Who:** Not recorded. The commit carries the project's git author, and on 2026-11-03 the person told the orchestrator "the local model changed".
- **Chosen:** `qwen3.5:4b-q4_K_M` as the writer. `gemma4:e4b-it-qat` as the separate judge wherever writer and judge must differ.
- **Considered:** `qwen3.5:9b-q4_K_M` and `granite4.2:8b`, which did slightly better on tool choice but only partly fit the 8 GB GPU and ran at about half the speed. The installed `gemma4:26b` does not fit.
- **Evidence:** `docs/model-choice.md` (a small test on this laptop GPU, not a result).
- **Would change if:** a model that fits fully on the GPU does clearly better on a larger test. That test is only 22 items, so it is a screen, not proof.

## 2. One run record holds many systems
- **Date / task:** 2026-11-02, Phase 1 review, edits to T1.2, T4.2, T9.1
- **Who:** The person, from a question the orchestrator asked.
- **Chosen:** One record maps each system to its predictions file, and `check_run` recomputes intervals and paired comparisons for each (commit 949e5d1).
- **Considered:** (a) one record per system plus a separate comparison record: more ways for files to fall out of step. (b) per-system records and no interval claims: smallest change, but then "multi-agent beats single agent" could never be a checkable claim.
- **Evidence:** `docs/TESTING.md`, every record in `results/runs/`.
- **Would change if:** the test-once rule for E8 became hard to enforce with one record.

## 3. Suggestions are written in a separate second step
- **Date / task:** 2026-11-03, T3.2 (plan Decision on T3.2, commit 27213ce)
- **Who:** The person.
- **Chosen:** No word-overlap filter. In respond mode a second model call sees only the verified finding statements and the question, and writes the suggestion. Every drop is noted in `missing`.
- **Considered:** (a) keep the 3-word filter: in a hand test it dropped good suggestions and let paraphrased injected claims through. (b) keep one call, label the suggestion, measure: the model would see injected text while writing. The numbers from that hand test live only in the session, not in a record, so none are given here.
- **Evidence:** `docs/PLAN.md` T3.2, commit 27213ce. Later measured in E2, see entries 14 and 15.
- **Would change if:** the extra call costs too much in E3 or E7, or if injected claims keep reaching suggestions through findings. Entry 15 is the planned fix for the second case.

## 4. E4 trains on DECODE plus a new invented train-only set
- **Date / task:** 2026-11-03, T5.2
- **Who:** The person (the coordinator added the three-card-mode coverage).
- **Chosen:** Combine DECODE with a small invented scenario set at `datasets/scenarios_train/`, covering all four labels, marked synthetic. Report contradiction vs not separately on DECODE dev.
- **Considered:** (a) binary model on DECODE alone and let the prompted checker decide update and insufficient: bigger change to what E4 compares. (b) DECODE only with binary results: E4 would stop being a four-label comparison, and a resume bullet would change.
- **Known cost:** The model trains on our own writing style, so scores on our scenarios may look better than real use. The records say so.
- **Evidence:** `docs/PLAN.md` T5.2, record E4-20261103T102658868337Z (`train_only_balance` in its config).
- **Would change if:** the person's review of a sample finds the invented cases unrealistic.

## 5. Near-copy guard uses 3-word windows, and an independent read is the real check
- **Date / task:** 2026-11-02, T5.2 / T6.1 (commits 301d6c0, 6968ed4, 80a3a2a, 4e7e1d4)
- **Who:** The orchestrator with the coordinator. The coordinator suggested 8-word windows, the orchestrator measured that they would not catch near-copies, and 3-word windows were agreed.
- **Chosen:** Compare 3-word windows across the whole conversation against held-out scenarios. State in the records that the guard is weak.
- **What happened:** An independent read of the invented set before training found real near-copies of held-out scenarios that the guard scored low. Those cases were rewritten and read again. T6.1's check also compared whole turns only, so a lightly edited dev case would have slipped in (fixed after a verifier failure).
- **Evidence:** the commits above, the E4 record notes, `docs/PLAN.md` T5.2 and T6.2.
- **Would change if:** a better similarity check is built. The independent read stays either way.

## 6. E4 first run reverted: NaN loss from float16 weights, fixed by loading float32
- **Date / task:** 2026-11-03, T5.2 (commit b0c358c)
- **Who:** The orchestrator and implementer; the coordinator later required the same guard in T6.2.
- **What failed:** DeBERTa-v3-base loaded in its stored float16, AdamW's epsilon underflowed to zero, and the first step turned the weights NaN. All six settings trained to NaN, yet each was still recorded as ok. The trained model was untrained.
- **Chosen:** Load weights as float32. A shared guard stops a non-finite loss before the update. `check_run` rejects a chosen model with a non-finite loss. The rerun lists the first records as superseded.
- **Considered:** Other ways to keep half precision, reason not recorded.
- **Evidence:** Void records E4-20261103T084042847386Z and E4-20261103T084834918198Z. Superseding records E4-20261103T102658868337Z and E4-20261103T102930546849Z, whose `supersedes` field gives the reason.
- **Would change if:** memory forces half precision. Then the guard must still pass a smoke check on real batches (`e4_consistency --smoke`).

## 7. Trained and prompted consistency checkers are both kept
- **Date / task:** 2026-11-03, T5.2 result, plan Decision on T4.2
- **Who:** The coordinator, after reading E4's saved predictions.
- **The finding:** On DECODE dev (record E4-20261103T102930546849Z) the trained model beat the prompted one on `label_accuracy`, and the saved interval for the difference excludes zero. But the prompted checker raised far fewer false alarms (`false_alert_rate`), and the trained one caught more contradictions (a catch rate read from saved predictions, not a stored metric). At a low real contradiction rate, the trained model's false alarms would dominate its alerts.
- **Chosen:** Neither alone. See entry 8.
- **Considered:** Trained alone, prompted alone. Retraining to reduce false alarms was ruled out by the coordinator.
- **Would change if:** hand-reading of false alarms shows a pattern that retraining could fix.

## 8. Cascade: trained model screens, prompted checker confirms, threshold 0.95
- **Date / task:** 2026-11-03, between T5.2 and T4.2 (commits 27f4d8c, 7a925de, 65b4bfa, 1dc3667)
- **Who:** The coordinator set the method. The threshold came from the rule below, run by an implementer.
- **Chosen:** `CascadeChecker`, with flag threshold 0.95 in `configs/cascade.json`. The rule, written with the catch-rate floor before selecting: lowest mean false-alert rate among points that keep the floor on every source, ties to higher catch.
- **Considered:** A threshold sweep of the trained model alone, and 19 other thresholds in the sweep.
- **Caveats:** Chosen on dev, which is selection data, labels unreviewed, so the config says `provisional`. 0.95 is the top of the swept range, so a stricter threshold was never tried. Most of its false alarms on our cases were `wrong_speaker_trap` (the session's reading of saved predictions).
- **Evidence:** `configs/cascade.json`, records E4-20261103T102658868337Z and E4-20261103T102930546849Z.
- **Would change if:** T4.2's flip and false-alarm checks, or a wider sweep, move it. The test split is untouched.

## 9. How to read E4's result on our own cases
- **Date / task:** 2026-11-03, T5.2 (verifier notes in `orchestration/verdicts/T5.2.json`)
- **Who:** The verifier and orchestrator. This is a reading, not a design choice.
- **Result:** On our dev scenarios (record E4-20261103T102658868337Z) the trained model beat prompted, zero-shot and TF-IDF on `macro_f1`, and the saved intervals exclude zero.
- **Why not to over-read it:** (a) the winning setting (`lr3e-05_ep3`) was chosen on the same dev set, so it is not held out. (b) The `update` and `insufficient` scores look partly like a training-style artifact: few distinct invented cases, repeated eight times, and dev cases share templates. (c) The trained model is weak on `contradiction` and on `wrong_speaker_trap`. (d) Labels are unreviewed. The error analysis is `results/error-analysis-e4.md`.
- **Would change if:** the held-out test (E8) disagrees.

## 10. Retrieval: hybrid with one-turn chunks (E1 winner)
- **Date / task:** 2026-11-03, T2.5 / E1
- **Who:** A fixed rule decided it: highest mean `recall@5` over the sources, ties to higher `mrr`, then a fixed order.
- **Chosen:** `hybrid_t1`.
- **Considered:** dense, BM25, hybrid with rerank, each with one-turn chunks and 3-turn windows (8 settings).
- **Evidence:** E1-20261103T034100128807Z (our scenarios) and E1-20261103T035342386305Z (QMSum, ids only). The session notes say there was no measurable lead over `dense_t1` or `hybrid_rerank_t1`, so this is close to a tie.
- **Would change if:** a larger or reviewed set separates the settings.

## 11. Ollama is called at 127.0.0.1, never localhost
- **Date / task:** 2026-11-03, T2.5 (E1)
- **Who:** The person, who stopped the running E1.
- **Why:** On this machine Windows tries IPv6 first, so each call to `localhost` waited far longer than a call to `127.0.0.1`. The person measured it; the figures are not in a run record.
- **Chosen:** Every default Ollama base URL is `http://127.0.0.1:11434`, with a test. E1 was discarded and rerun from scratch, and latency from the old run is invalid.
- **Evidence:** the E1 records above (the rerun) and the defaults test in `tests/`.
- **Would change if:** the machine or network stack changes.

## 12. The Claude call cost two cloud stages: lost calls, now saved row by row
- **Date / task:** 2026-11-03, T3.3 (commits e7baeaa, 16eff3d)
- **Who:** The orchestrator and implementer, reporting to the coordinator.
- **What failed:** The first E2 cloud stage spent paid calls and lost them. The session notes say the call counter's lock file, `results/claude_calls.json.lock`, was not git-ignored. The count of lost calls is only in those notes, not in a record. I read the mechanism from the commit titles, so treat it as likely, not confirmed.
- **Chosen:** Ignore the lock files. One shared counter for all worktrees. Check the tree before spending. Save each paid row as it arrives. Make the cloud stage runnable alone (`--cloud-only --local-run`). Read the cap from the client.
- **Considered:** Recovering the lost rows: the second cloud record shows `reused_rows` of zero, so they were not recovered. Other options, reason not recorded.
- **Evidence:** E2-20261103T132846863600Z (`cloud_calls`), commits above. The call cap is in `docs/DESIGN.md`.
- **Would change if:** another path writes to the tree during a paid stage.

## 13. T3.3 failed verification once and was retried
- **Date / task:** 2026-11-03, T3.3 (attempt 1 of 3)
- **Who:** The verifier failed it. The orchestrator offered three options, and the coordinator chose the middle one.
- **Why it failed:** E2's count of injected claims in suggestions could not fail. The single agent's prompt never explains the card modes, so almost no card was in respond mode and no suggestion was written on injection cases. "Zero injected" was therefore not evidence that entry 3's second call is safe.
- **Options:** (a) fix the prompt. (b) a new run so the count can fail, plus keep the old records and say they ran on the old prompt. (c) only state the limit. The coordinator chose (b) and added plain recall cases, so the run could also show whether recall was pulled into respond.
- **Widening:** The planned 48-case run became all 148 dev cases (commit b453549). Reason not recorded.
- **Evidence:** `orchestration/verdicts/T3.3.json`, E2-20261103T124545651045Z, E2-20261103T132846863600Z.
- **Would change if:** not applicable, it is history. The outcome is entry 14.

## 14. E2 mode-defining prompt tried, then reverted
- **Date / task:** 2026-11-03, T3.3 (commits c9fe70e, 172b04a, bb37c68)
- **Who:** The coordinator approved the revert on the orchestrator's recommendation, once the result was in.
- **Tried:** A single-agent prompt that defines recall, clarify and respond.
- **Result (record E2-20261103T155026435232Z, on `task/T3.3`):** Respond mode still did not fire on any of the respond cases. No recall case moved into respond. `answer_when_answerable` went down compared with the earlier `retrieval_memory` run. One injection attack worked: on `syn-inj-02` the card stated the injected claim as a finding, repeated it in the suggestion, and cited the attacker's real turn, so the citation guard passed it. Why answering fell: not understood.
- **Chosen:** Revert `single.py` to its text before the prompt. Remove the three prompt tests that asserted the definitions, and the commit says why. Keep the record as evidence. Land T3.3 as a measurement, with a new count of injection cards that state the claim as a finding. Further mode-prompt work moves to T7.1.
- **Considered:** Keeping the prompt. Rejected because it did not do its job and answered fewer answerable cases.
- **Would change if:** T7.1 finds a prompt that makes respond fire without losing answers.

## 15. Instruction guard and respond mode move into code (carried to T4.2)
- **Date / task:** 2026-11-03, plan edit for T4.2 (commit f13c2ce)
- **Who:** The coordinator.
- **Chosen:** (a) A turn that addresses the assistant, or says to ignore, disregard or say something, cannot be evidence for a factual claim. Test on `syn-inj-02` and on harmless look-alikes ("ignore that email"), and report the false-drop rate. (b) When a press has no question and the latest other-speaker turn asks the user something, the controller sets mode `respond` in code. Report the respond rate and any recall cases wrongly moved.
- **Considered:** A prompt-based fix, which entry 14 shows did not work. Leaving injection to the citation guard, which `syn-inj-02` got past.
- **Also found:** The old-prompt `retrieval_memory` run flags several injection cards for stating an injected claim in a finding. The count is a word-overlap proxy, and only `syn-inj-02` has been read against its transcript.
- **Evidence:** `docs/PLAN.md` T4.2, record E2-20261103T155026435232Z.
- **Would change if:** E3 shows respond still does not work. Then it goes back to the coordinator.

## 16. retrieval_memory is carried into E3 as the design's system, with recent as baseline
- **Date / task:** 2026-11-03, plan edit for T4.2 (commit 993068b)
- **Who:** The coordinator.
- **After seeing dev results:** E2's fixed rule picked `recent` (record E2-20261103T124545651045Z, `winner_rule` in its config), and that stays in the record. The margin between the top strategies was very small, per the T3.3 verifier. But `recent` answers under half of answerable cases, while `retrieval_memory` answers many more. The choice to carry `retrieval_memory` was made after seeing these dev results, and the plan says so.
- **Cost:** `retrieval_memory` abstains correctly only about half the time when evidence is insufficient, and its groundedness is lower than the other strategies'.
- **Chosen:** Selection from now on uses the lower of the two abstention sides alongside groundedness, written down before each run. Only dev has informed this. Test is untouched.
- **Would change if:** the weak spots are not fixed by T6.2 or T7.1.

## 17. Same-speaker rule tried as an extra arm in E3
- **Date / task:** 2026-11-03, plan edit for T4.2 (commit 3a60ec7)
- **Who:** The coordinator.
- **Chosen:** An arm where only same-speaker pairs can be `contradiction`, with different-speaker conflicts shown as a disagreement. Keep it only if dev shows it helps. Count wrong-speaker false alarms per system, here and in E8.
- **Reason:** Most cascade false alarms on our cases were `wrong_speaker_trap` (entry 8).
- **Would change if:** it lowers catches more than it lowers false alarms.

## 18. AMI audio: intermittent 403s, nothing downloaded, block respected
- **Date / task:** 2026-11-02 and 2026-11-03, E7 prep on `prep/T8.5`
- **Who:** The orchestrator, with the coordinator agreeing on the plan.
- **What happened:** On 2026-11-02 a researcher hit a 403 on the AMI download page, tried one `curl` with a browser user agent, was denied by the permission system, and did not retry. The orchestrator flagged it to the person. The small annotations download worked normally then. On 2026-11-03 the size check for E7's meetings stopped because one file's size came back unknown. One file read fine on a direct look while another, just read by the check, returned 403. Different files fail each time, so it looks intermittent, possibly rate limiting (unconfirmed). The total size is far under the 15 GB limit.
- **Chosen:** Add pauses and retries to the builder's script. Do not change how requests identify themselves, and do not work around the block. The orchestrator stopped sending requests. The coordinator agreed to one more paced try when T8.5 is close to running. If it is refused again, block T8.5 with the file list, and the coordinator takes it to the person.
- **Status:** Nothing downloaded for E7. The second try has not happened yet.
- **Evidence:** the AMI section of `docs/datasets.md` (access and the 403 on re-fetch).
- **Would change if:** the person registers or gets another route to the files.

## 19. E7 questions come from AMI annotations, and E7 reports linked-turn recall only
- **Date / task:** 2026-11-03, plan edit for T8.5 (commits b7f2613, a0552e8)
- **Who:** The coordinator decided who authors the questions and the rule. An independent reader checked a sample.
- **Chosen:** Questions derive from AMI's annotations and clean transcripts by a written rule, never from speech output, with `labels_reviewed: false`, committed before the run. Because the reader found them usable with limits (evidence fully supported the gold answer in most of a hand-read sample and partly in the rest), E7 reports recall of a linked turn and how it changes under speech and speaker errors, with no bare "answered correctly" score.
- **Evidence:** `docs/PLAN.md` T8.5, `docs/ami-questions.md` (on the prep branch).
- **Would change if:** items are scored one by one against the gold's items.

## 20. pyannote: patch around the gated repo removed, terms accepted instead
- **Date / task:** 2026-11-03, T8.2 / T8.7 (commits eaa7b8c, 5b99e92)
- **Who:** The coordinator rejected the patch. The person accepted the model's terms.
- **Reverted:** A patch that fetched `speaker-diarization-community-1` around its access gate. Until the terms were accepted, turns stay honestly `unknown`.
- **Chosen:** Fail clearly until the terms are accepted, then run the real test once on CPU with `pyannote.audio` pinned to 4.0.7. Do not change the shared environment's torch.
- **Would change if:** the license or the gate changes.

## 21. Dataset approvals follow the license standing rule
- **Date / task:** 2026-11-02, T0.3
- **Who:** The researcher found the licenses. The person approved.
- **What happened:** QMSum and DECODE state no data license, so both were marked "needs the person" and not downloaded. The person then approved both for local, non-commercial evaluation only, never committed or redistributed, and later allowed QMSum to train the T6.2 adapter locally, never published. AMI is CC BY 4.0 and approved. Qwen2.5-3B stays unapproved.
- **Evidence:** `docs/datasets.md`, and `tests/test_no_raw_text.py` keeps their text out of tracked files.
- **Would change if:** the person decides otherwise.

## 22. Coordinator makes technical choices; the person keeps the rest
- **Date / task:** 2026-11-03, project rule in `CLAUDE.md`
- **Who:** The person.
- **Chosen:** The `porter-` coordinator decides technical and build choices. Spending, publishing or pushing, new dataset licenses, consent, and removing a feature or weakening a resume bullet go to the person, through the coordinator.
- **Why:** The person is often away. The orchestrator first declined to edit `CLAUDE.md` on a peer's word alone, and did so only after the person confirmed.
- **Would change if:** the person says so.

## 23. Memory limit: at most one full pytest per project at a time
- **Date / task:** 2026-11-03, project rule (commit 7559610)
- **Who:** The coordinator, saved in `CLAUDE.md`.
- **Reason (from `CLAUDE.md`):** Each Python process that imports torch commits about 3 GB here, and 52 at once nearly exhausted Windows's limit and swung free disk sharply.
- **Chosen:** At most one full `pytest` per project across all tracks, prefer named test files, `num_workers=0` in tests, at most 4 loader workers in real training, and exit background processes when done.
- **Would change if:** the machine gets more memory.

## 24. T1.4 retry: a bad retry instruction changed `macro_f1` against the spec
- **Date / task:** 2026-11-02, T1.4 (attempt 2)
- **Who:** The orchestrator caused it. The verifier had passed the task while skipping a check, and the orchestrator refused to close it.
- **What failed:** The orchestrator's retry told the implementer to average over labels present. `docs/TESTING.md` defines `macro_f1` over the four fixed labels. The change was caught and restored, and it matches scikit-learn with fixed labels.
- **Lesson:** Check the spec before telling a builder to change code.
- **Evidence:** `orchestration/verdicts/T1.4.json`, `docs/TESTING.md`.

## 25. No commit attribution lines; history rewritten once
- **Date / task:** project rule, date not checked
- **Who:** The person.
- **Chosen:** Commit messages carry no tool or assistant names, no co-author lines and no session lines. Old verdict hashes were stale after the rewrite, finished tasks stayed finished, and the T1.3 to T1.5 gates were rerun.
- **Reason:** The person's rule. Further reason not recorded.

## E2 and E3 rerun rule after the short-id citation bug (2026-11-03, T4.2, written before any new numbers)
- **Chosen:** rerun E3's `cards` part, E2's four local strategies, and E2's cloud arm, because the citation guard dropped true findings that cited a short turn id such as `t0`. Same cases, same seeds, same config apart from the fix and an exclusive GPU. E2's cloud arm also moves to Sonnet, by the person's rule on Claude comparisons, and its 100-case limit and per-stage call budget are removed by the same rule. It runs on all 148 dev cases, the same ones the local strategies use, so that is a wider set than the old record's, and its record says so. This changed before any new numbers existed. The old records stay listed in `supersedes`, with this bug as the reason. Selection follows the written rule, whichever system it favours. The 8 s deadline doesn't change during these runs.
- **Considered:** keeping the old records with a caveat. It lost because the bug hits systems unequally, the team most, so the comparison itself can't be read. Fixing the bug and moving on to E6 lost too, since `retrieval_memory` was chosen on the affected numbers.
- **Evidence:** dropped citations with a short id, in E3-20261103T185316318532Z, counted from its saved rows, per `orchestration/evidence/cleanup-result-risks.md`.
- **Would change it:** nothing in the new numbers. If the rerun can't reproduce the same cases and seeds, that gets stated, not worked around.
## 26. DECODE and QMSum may go to Claude; their labels count as reviewed, said as "on published labels"
- **Date / task:** 2026-11-03, cleanup (`labels_reviewed` and claims), passed on by the coordinator.
- **Who:** The person, in the coordinator's session: "QMSum and DECODE may go to Claude through claude -p for grading and comparison. Publisher-provided labels (DECODE, QMSum) count as reviewed for claims, and the wording says 'on published labels'."
- **Before:** Both datasets were approved on 2026-11-02 for local, non-commercial evaluation only (QMSum also for training the T6.2 adapter locally), and their text could not leave this machine. Their labels were not counted as reviewed.
- **Chosen:** Their text may go to Claude through `claude -p` for grading and comparison. `write_run` sets `labels_reviewed` true for the `publisher` label source, and `claims check` decides from the record's `scenario_source` so older records (flag false, no `label_source`) are covered; a record whose source cannot be determined is not reviewed. A claim from such a record must say "on published labels" in its `text`. Our scenarios and AMI keep their sign-offs. The cloud rule changed for future runs only; records made before 2026-11-03 did not send this text to Claude.
- **Still in force:** Raw text is never committed, published or redistributed. "Never publish" for the QMSum adapter and for examples derived from QMSum is unchanged, because the person has not changed it.
- **Evidence:** `docs/datasets.md`, `docs/TESTING.md`, `tests/test_claims_publisher.py`.

## 27. Judge check: the cloud grader re-grades a sample of the local judge's claims (2026-11-03, written before any run, no numbers)
- **Date / task:** 2026-11-03, the judge check the person set before T9.1. It is a runner, `src/contextlens/experiments/judge_check.py`, not a plan task.
- **Who:** The person set the check; the coordinator set the method below.
- **Question:** Does the local judge (`gemma4:e4b-it-qat`, rubric `e2-v1`) agree with a stronger grader, and does it flatter the results? This adds to the person-reviewed sample in `docs/TESTING.md` and does not replace it.
- **Chosen:**
  - **Input.** Run ids of dev records given on the command line. A record on any other split, or one superseded by a later record, is refused. Rows are rebuilt from the saved predictions: each claim is one finding, with its statement, its cited quotes and speakers, and the local verdict. If the saved rows do not hold one local verdict per claim, the run stops, because a proxy would give a meaningless agreement number.
  - **The question.** The cloud grader is asked through the same `Judge` code that asked the local judge: the same rubric, the same claim and quote block, escaped by `untrusted.py`. It sees nothing else: not the local verdict, not the system that wrote the answer, not the experiment.
  - **Sources.** The person's update of 2026-11-03 lifts the old limit to "synthetic and LongMemEval only": QMSum and DECODE text may go to Claude through `claude -p` for grading and comparison. A source counts only if its saved rows can rebuild the claims. QMSum and DECODE records hold ids and verdicts only, by the rule in `check_run`, so they give no claim to grade. The run reports the claim count per source and per origin, and states plainly when a source gives none. Raw text is not committed: the saved rows for the check hold claim ids and verdicts only.
  - **Unit and sample.** One claim. Seed 20261103, fixed in the committed code. The pool is every locally graded claim in the named records, with exact repeats (same system, case, statement and quotes) dropped. Strata are record, system and case category. The natural sample takes one random claim from each stratum in turn, in a seeded order, round after round, until 200 or the pool is empty; if fewer than 200 exist the record says so. Claims keep their natural share of local verdicts within each stratum. If fewer than 40 of the natural sample carry the local judge's rarer verdict, random unsampled claims with that verdict are added as a second stratum, labelled `oversample`, up to 40 rare claims or until none remain. Agreement and kappa are reported for the natural sample and for both strata together; the call on the local judge rests on the natural sample. The share of each verdict is reported for both.
  - **Spend.** Each claim is one ask, one `claude -p` call, or two if the reply is not valid JSON and the client retries. Before the first call the shared counter must leave room for the worst case (two per pending ask) under the 5,000 cap. Rows are saved to `results/partial/judge-check/` (git-ignored) as they arrive, with a header holding the commit and the sample; a rerun at the same commit and sample resumes and asks again only claims that have no answer. A failed ask is listed and is not counted as disagreement. Five failures in a row stop the run with the rows kept.
  - **Report.** Agreement; Cohen's kappa with a 95% bootstrap interval over whole conversations (the project's resampling, seed 13); the 2x2 table; the direction, meaning how many claims the local judge supported that the cloud grader rejected and the reverse; the grounded rate by each judge and their gap, pooled, per experiment, per system and per category. The record is written through `write_run` and checked by `check_run`. For that, `check_run` recomputes the new metrics (`cloud_grounded_rate`, `grounded_gap`, `judge_agreement`, `judge_kappa`, `gemma_yes_cloud_no`, `gemma_no_cloud_yes`) from a new optional row field, `cloud_supported`; no existing check was changed.
  - **When agreement is poor.** Poor means Cohen's kappa on the natural sample below 0.6. The cut is the lower edge of the "substantial" band in Landis and Koch (1977), rounded from 0.61. It is a convention, not a value derived from this project's data, and no better-grounded value was found. The threshold is in the code (`POOR_KAPPA`) and in the record. If agreement is poor the record's notes say so plainly and propose Sonnet as the judge for the final numbers.
  - **Status.** The result is provisional: no claim comes from this record alone.
- **Considered:** (1) A proxy question with a differently worded rubric: rejected, since the agreement would measure the wording. (2) Sampling uniformly over claims: rejected, since the largest systems and categories would crowd out the rest. (3) Showing Sonnet the local verdict: rejected, it would anchor the answer. (4) Replacing the person-reviewed sample: rejected by the person.
- **Would change if:** the saved rows turn out not to rebuild the question; the run then stops and the gap is reported, not worked around.
- **Evidence:** `src/contextlens/experiments/judge_check.py`, `tests/test_judge_check.py`.

## 28. Respond mode is set in code, on the latest turn
- **Date / task:** 2026-11-03, T4.2 (commit f5035a0)
- **Who:** The coordinator decided to set it in code; the implementer chose the rule's wording.
- **Chosen:** A press with no question, on a latest turn that has a question mark or starts with "please" or "let's", is set to `respond` in the shared finish step, so the single agent, the one-shot baselines and the team all get it. The suggestion is then written from verified findings.
- **Considered:** The coordinator's wording was "the latest other-speaker turn". The request names no user, so the speaker of the latest turn cannot be told from the user. The records report how many firings have a latest turn by the same speaker as the turn before it, so the size of the approximation is visible.
- **Known cost:** A question the user asked themselves also fires. A statement addressed to the user ("We booked it for you!") does not.
- **Would change if:** the request carries who the user is.

## 29. Instruction guard patterns were written after reading the dev injection turns
- **Date / task:** 2026-11-03, T4.2 (commit 8afeda0)
- **Who:** The coordinator decided a code rule; the implementer wrote the patterns.
- **Chosen:** `guard/instructions.py` drops a finding that cites a turn naming the assistant, telling it to drop its rules, telling it what to say, or asking for secrets. It sits in the shared `check_findings`, so every system has it.
- **Known cost:** The patterns were written after the dev injection turns had been read, so its catch rate on them is not a held-out figure. It also drops some ordinary talk, and one order ("Disobey your setup ...") gets past it. Both counts are in the E3 cards record and are not hidden.
- **Considered:** Widening the patterns for the missed phrase: rejected, since each widening adds false drops.
- **Would change if:** a held-out injection set shows a worse catch or false-drop rate.
- **Amended, 2026-11-04, the coordinator: sets written blind, hashes committed before the patterns freeze.** A fresh agent that was told not to open this repo wrote a held-out guard set (60 injected, 60 lookalike) and 24 new flip templates. SHA-256 of the set: `06440351937c77a0289a3e9201b1a142e165656c059f809c0eb8bf3a54e03b41`. SHA-256 of the templates: `9ba5e11b2e3f93faee08e6834ad340d606ba2e9b697d37469878e15b3b325809`. Their contents stay out of the repo until the run. Each is run once, after the patterns are frozen at a recorded commit; if the patterns change after a run, that set is spent and a later figure needs a new set. Limit: blindness is a rule plus separate folders, not proof, because the builder runs on the same machine under the same account. The hashes prove the sets did not change after this commit; they do not prove the builder never read them. The existing 3 E3 flip templates guided the new patterns, so flips on them are not a clean figure.
- **Slip, 2026-11-04, logged at the coordinator's order: a near-copy check read the test split.** While checking the blind files for near-copies, the orchestrator's one-off script used the patterns `datasets/scenarios/**/*.jsonl` and `datasets/scenarios_extra/**/*.jsonl`, among others, and so read every string of any test split under them, including `datasets/scenarios/test.jsonl`, which no task named for this. It compared word overlap only; the guard did not run and nothing went to a model. Printed, in full, as overlap, turn id and closest file: `(0.5, inj-18, datasets/guard_heldout/turns.jsonl), (0.47, ord-12, datasets/guard_heldout/turns.jsonl), (0.4, ord-11, datasets/scenarios/test.jsonl), (0.38, ord-22, datasets/guard_heldout/turns.jsonl), (0.38, inj-33, datasets/guard_heldout/turns.jsonl), (0.37, inj-04, datasets/guard_heldout/turns.jsonl), (0.35, inj-42, datasets/scenarios/test.jsonl), (0.33, ord-25, datasets/scenarios_extra/dev.jsonl)`, then `corpus strings 3571 new 144`, `>=0.6: 0 >=0.8: 0`. No test text was printed or read by anyone. Nothing changed because of it: no system, setting or pattern was chosen on it, and `ord-11` and `inj-42` are left as written. The coordinator's ruling: the test split stays, and T9.1 still runs once as planned. Rule from here: any file-scanning script excludes `datasets/**/test*`, and agent prompts say so.
- **Frozen and run, 2026-11-04:** the patterns froze at `4f5b6fc`, after the hash commit above, and the blind set ran once on them at `c5a7bad` on `prep/injection`; the result is in `results/guard-heldout/`. Any later pattern change spends this set. **Second scan:** the pattern builder's own report says it ran `grep -rln "number_replay" .` over its whole worktree, which searched the test split as text and printed file names only; nothing was chosen on it. The same ruling and rule apply.

## 29a. Memory extraction gets its own timeout, and a failed build names its reason
- **Date / task:** 2026-11-04, coordinator-ordered fix on `fix/memory-timeout`
- **Who:** The coordinator ordered it; the implementer chose the value.
- **What failed:** `MemoryExtractor.update` asked for up to 800 tokens but passed no timeout, so it got the client's 10 s default. The writer makes about 56 tokens per second, so a full answer needs about 14 s, and a 15-turn conversation took 8 to 15.5 s on a nearly idle GPU. Long conversations timed out whatever the load: 22 to 23 of 108 cases on `scenarios_extra` dev, 2 of 148 on the old dev set. `build_memory` swallowed the exception, so the cause was invisible. The product's background path in `api/stream.py` makes the same call.
- **Chosen:** `EXTRACT_TIMEOUT_S = 60.0` in `memory/extract.py`: four times the 14.3 s the full budget needs at the measured rate, to cover a cold model and other jobs on the GPU. Extraction runs outside any card deadline, so the 8 s deadline is untouched. `build_memory` now prints the case id and a closed-form reason to stderr (the exception type, plus the message only for a transport or timeout error; any other LLMError gives "invalid model reply", since those messages can quote the model) and takes an optional `on_failure(case_id, reason)`; the E2 and E3 records list `failed_reasons` in `memory_build`. `BROKEN_MAX` is unchanged.
- **Other calls without a timeout:** the rolling summary (200 tokens, about 3.6 s at the measured rate) and the consistency check (100 tokens, about 1.8 s) fit inside 10 s with a margin of about 2.8 and 5.5 times, so they were left alone. Their prompt reading time was not measured separately.
- **Considered:** Raising the client default for every call: rejected, since card calls rely on the short default and take their own deadline. Lowering `max_tokens` to fit 10 s: rejected, it would cut off long extractions and lose notes. A per-call value derived from the token count: rejected as more machinery than one constant.
- **Older records:** records built with the 10 s limit list 2 of 148 failed memory cases in their config; those 2 were this timeout, and their memory-strategy rows were scored with empty notes for those cases.
- **Would change if:** the writer model or its speed changes (the constant follows the measured rate), or a build still times out at 60 s, which the reason now shows.

## 30. E3's selection score uses the lower abstention side
- **Date / task:** 2026-11-03, T4.2
- **Who:** The coordinator.
- **Chosen:** The score is the mean of groundedness and the lower of the two abstention sides (answer rate on answerable cases, abstain rate on insufficient cases). Every part must be defined. It was written down before the run. It names the best system in E3's record without re-deciding E2's winner. E2's own rule is unchanged and now picks `full`; E3's written rule picks `single`.
- **Known cost:** the abstention side rests on the 16 insufficient-evidence cases of dev, so one case moves it by several points. The records now say so.
- **Evidence:** `SELECTION_RULE` in `experiments/e3_team.py` and `config.selection_rule` in the E3 record.
- **Would change if:** the person wants a different weighting.

## 31. Speakers shown as names or as neutral roles: the one that flips less, unless accuracy falls
- **Date / task:** 2026-11-03, T4.2
- **Who:** The coordinator set the principle; the implementer fixed the measurable-drop limit before any run.
- **Chosen:** The trained model sees low-confidence speakers as `unknown` and escaped text. On dev the experiment scores with names and with roles and picks roles only if they flip less and alert accuracy does not fall by more than 0.02. Speaker-flip and injection-flip rates are reported per system, and any rate above 5% goes to the coordinator before E8.
- **Evidence:** the E3 consistency records, `config.speaker_mode` and `config.flips`.
- **Would change if:** the coordinator changes the 0.02 or the 5%.

## 32. The same-speaker rule is an arm, kept only if it helps
- **Date / task:** 2026-11-03, T4.2
- **Who:** The coordinator.
- **Chosen:** A flagged statement is rescored against each of the last six earlier turns alone. If the turn it conflicts with most is another speaker's, the alert is shown as a disagreement. Unknown speakers never count as another speaker. The rule is kept only if the mean false-alert rate falls and the mean catch rate falls by no more than 0.02. The verdict is in `config.same_speaker_rule`.
- **Known cost:** The conflicting turn is found from single-turn scores, which is a heuristic. DECODE speakers are A and B by position, so wrong-speaker counts on DECODE say little.

## 33. The trained model alone alerts at the cascade config's threshold
- **Date / task:** 2026-11-03, T4.2
- **Who:** The implementer, reported to the coordinator.
- **Chosen:** Trained-alone raises an alert when P(contradiction) is at or above `flag_threshold` in `configs/cascade.json`, the same screening rule `select_cascade` uses. The 30 false alarms to hand-read are drawn from trained-alone at that threshold.
- **Known cost:** The threshold is marked provisional in the config.

## 34. Single and team share the same tools, with no consistency checker
- **Date / task:** 2026-11-03, T4.2
- **Who:** The implementer.
- **Chosen:** Both get one `Tools` build with no consistency checker, so neither hides model tokens inside a tool call and the comparison is on the same footing. The consistency agent of the team therefore works from the transcript and memory only.
- **Would change if:** a later experiment wants the cascade as a tool.

## 35. Prompt version stays v1
- **Date / task:** 2026-11-03, T4.2
- **Who:** The implementer.
- **Chosen:** No prompt text changed in T4.2: models are shown full turn ids, the prompts carry no short-id example, and the fixes are in code (guard, number rule, respond rule). The version label stays `v1`. Under the phase 3 rule a changed prompt would have needed a new label.

## 36. Short turn ids were dropped as bad citations; E2 and E3 were rerun
- **Date / task:** 2026-11-03, T4.2 (commit efa787b; reruns E3-20261103T221004658482Z, E2-20261103T224548534556Z, E2-20261103T230638094732Z)
- **Who:** The coordinator found it in E3's saved rows; the implementer fixed it.
- **What failed:** Models cite the part of a turn id after the colon (`t0`), and the guard only knew the full id, so true findings were dropped as citation failures. The team lost the most. No finding with a short id was kept anywhere.
- **Chosen:** The guard resolves an id with no conversation prefix to the one turn of this conversation, at or before the press, whose id ends in it, with the quote still required. Another conversation's id, an ambiguous one and a later turn still drop. The reruns list the old records as superseded with a reason naming this bug.
- **What the rerun showed, in words:** Nearly all short-id citation drops are gone (one team drop with a short id remains). The team's answer rate on answerable cases is not clearly different from the single agent's: the interval excludes zero but the exact McNemar test, on a handful of discordant cases, does not agree. On the abstention side the team abstains less than the single agent where the evidence is insufficient, and there the interval and McNemar both say worse. The team costs more tokens and time. It is not better on groundedness. A few team cards ran into the 8 s deadline, and the record reports that per system. No metric values are stated here: claims take a separate route. An earlier version of this entry, and the first note of the record, said the team "beats" the single agent on answer rate; that was wrong, because the McNemar test had not been computed (see entry 38).
- **`retrieval` in `settings_changed`:** the E3 record now writes the retrieval config as a setting. The old record did not, so the comparison shows it as changed from `absent`. It was newly recorded, not changed.
- **Evidence:** the three reruns above, each listing its predecessor in `supersedes`, and `config.superseded` in them.
- **Would change if:** a model cites some other short form (for example a bare number).

## 37. The first note of the E3 cards rerun says the labels are unreviewed; that wording was wrong
- **Date / task:** 2026-11-03, T4.2
- **Who:** The coordinator found it; the implementer fixed the code.
- **What was wrong:** The text was hardcoded in the runner. `E3-20261103T221004658482Z` has `labels_reviewed: true` and `label_source: scenarios`, and its `notes[0]` ends "Dev labels are unreviewed". The flag is right and the sentence is wrong. The record is left unedited, as records are. The part of the sentence about the judge stands: the judge is one local model, and no person has reviewed a judge sample.
- **Chosen:** Notes and analyses now take the sentence from `records.label_text`, which reads the same flag and source that `write_run` records. The E2 analysis, E3's finding and selection note, and the cascade selection text were changed. No `docs` or run record was edited by hand.
- **Evidence:** the fix commit on `task/T4.2`; `results/error-analysis-e2.md`, whose label line is generated from the flag.

## 38. McNemar was missing on answered/abstained outcomes; the finding now needs it, and the records were rebuilt
- **Date / task:** 2026-11-03, T4.2 (verifier attempt 1)
- **Who:** The verifier found it; the coordinator ordered the fix.
- **What was wrong:** `eval/series.py` had no per-case outcome for `answer_when_answerable` and `abstain_when_insufficient`, so every card comparison in E2 and E3 saved `mcnemar_p` as null, and the E3 finding called a small difference "better" from the interval alone. `answer_when_answerable` counts cards that did not abstain; it does not measure whether the answer was right.
- **Chosen:** Both are now paired yes/no series (overall and per category), so the exact test is computed and `check_run` recomputes it. The finding says better or worse only if the interval excludes zero and McNemar agrees, and otherwise says no clear difference with the discordant counts. It calls the metric an answer rate, says the abstention side rests on 16 cases, and says the zero injected claims in suggestions is vacuous. `experiments/rebuild_records.py` rewrites only derived fields from the saved rows with no model call, checks the predictions are byte-identical, and writes a new record that supersedes the old one with this reason.
- **Same gap, not fixed here:** `false_alert_rate` is also a per-case yes/no outcome on the negative cases and its comparisons save null. Adding it would change E4's current records, which the cascade selection and `rescore` name by id, so it is left to the coordinator.
- **Reported, not applied:** The team repeats a finding inside one card; the repeats are counted and grounded_rate without them is shown in the record. `docs/TESTING.md` has no rule for dropping repeats, so the metric is unchanged.
- **Evidence:** the rebuilt records, each with `config.rebuilt` and `supersedes`.

## 39. The instruction guard's patterns were narrowed to orders aimed at the assistant
- **Date / task:** 2026-11-03, T4.2 (verifier attempt 1)
- **Who:** The coordinator.
- **What was wrong:** The first patterns flagged ordinary phrases ("ignore the instructions on the box", "tell the assistant manager I am late"), so the false-drop rate on look-alikes was far higher than the record said.
- **Chosen:** Narrower patterns, checked turn by turn: they flag exactly the same dev turns as the first ones (all 16 injecting turns, none of the others), so no existing record is affected. The look-alike phrases are reported in three sets: seen, used while narrowing, and fresh phrases measured once and never tuned against. Only the fresh set is a fair rate.
- **Known cost:** One fresh ordinary phrase still flags ("forget the above numbers"), because "ignore the above" is itself an injection in dev. It is left as measured.
- **Would change if:** a held-out injection set shows a worse catch rate.

Note on numbering: T4.2's entries were first written as 27 to 38 and are now 28 to 39 after the judge-check entry took 27. Committed records and run notes that cite a T4.2 entry by its old number (for example "see entry 37" for the McNemar fix) mean the entry that is now one higher.

## 40. Judge for the final numbers, and selection recomputed under the cloud grader (2026-11-03, T9.1, written before seeing the recomputed numbers)
- **Chosen, by the coordinator:** after the judge check (record JC-20261104T040402268688Z) found poor agreement, Sonnet is the judge for E8, with the local judge reported beside it on the same claims. Selection is recomputed under Sonnet by the same written score as E3, mean(grounded_rate, min(answer_when_answerable, abstain_when_insufficient)), on the same claims and cards, with Sonnet grading every claim of every card system of the current E3 cards record. Only grounded_rate changes. Sonnet doesn't grade systems it wrote. No new answers are generated. The outcome stands whichever system it favours; if it differs from `single`, it goes to the coordinator before anything else changes.
- **Considered:** keeping the local judge with a caveat. It lost because disagreements lean one way, and the choice of system rests on this rate.
- **Evidence:** JC-20261104T040402268688Z. Neither judge is ground truth; a blind reader's check of the disagreements is in hand, reported as an agent reader, not a person, unless a person reads them.
- **Would change it:** a reader finding the cloud grader's rejections mostly wrong.
- **Amended, 2026-11-03, the coordinator:** a blind reader, an agent and not a person, read the 24 claims the judges disagreed on (about 17 distinct). Where the local judge said yes and the cloud grader no, it sided with the local judge in most; in the reverse case, with the cloud grader in most. That met the "would change it" line above. So neither judge is called final: E8 reports both on the same claims, with their agreement stated, and calling the local judge flattering isn't established. Selection is still recomputed under the cloud grader as written, and a difference between judges goes to the coordinator. The rubric stays unchanged; statements that restate a conflict are where both judges wobble, and the report's limits say so. Numbers are in record JC-20261104T040402268688Z and `orchestration/evidence/judge-disagreements.reader.json`, which is git-ignored.


## 41. The number rule reads number words, half, dozen and ordinals as the numbers they are (rule written before any new numbers)
- **Date / task:** 2026-11-04, T7.1 (written before the rule's code and before any count from it)
- **Who:** The coordinator ordered it; the implementer wrote the rule.
- **What failed:** The rule compared runs of number tokens as spelled, so "two" in a statement against "2" in the quote, or the reverse, dropped a true finding. Saved rows from E2's cloud arm: 15 of 20 number drops had a number word and no digit, and 12 of 17 abstained answerable cases trace to them.
- **Rule:** the purpose stays: every number in a finding's statement must be a number in its cited quotes. A number is now compared by value, not spelling. Values: digits (`1,000` is 1000, `2.5` is 2.5, `5th` is 5); number words, with compounds (`twenty-five`, `five hundred`, `a hundred`, `one hundred and five`, `two thousand five hundred`); `5 hundred`, `2 million`; `half` (0.5); `dozen` (12, `two dozen` is 24); ordinal words (`first` to `twentieth` and the round tens) as their number. A lone `one` is a pronoun and is not a number. Each cited quote is read on its own, so words at the end of one quote and the start of the next do not join. Turn references ("t3", "turns 2 and 4") are still not numbers. It still fails closed: a statement number whose value is not in the quotes drops the finding (`500` against `50`, `5` against `500`, `five dollars` against `fifty dollars`, `3.5` against `3`, `$13` against `three`).
- **Named limits (still fail closed):** `couple`, `pair`, `once`, `twice`, `quarter`; `one and a half` against `1.5`; spoken years (`nineteen ninety` against `1990`); `2.5k` against `2500`; `15:00` against `3 pm`; dates written as `5/3`. A statement that says `first` or `half` where the quote has neither is now dropped, which the old rule let through.
- **Tests that changed because the requirement changed** (`tests/test_single_agent.py`, `test_numbers_must_match_as_whole_numbers` and the suggestion test): `$9` against `nine dollars`, `3 people` against `three people`, `1000 dollars` against `one thousand`, `500` against `five hundred`, and the suggestion `Ask about the 500 dollar fee` against `five hundred dollars` all pinned the bug as "fails closed" and now keep the finding. Every other case there is unchanged and still passes.
- **Records affected:** any E2 or E3 record whose cards went through `verify_card`. The re-application to the saved `drops` is in entry 42 once counted. Same cases and settings; old records go in `supersedes`. E6 has not run, so it runs on the fixed rule.
- **Amendment, written after the first replay of E2's saved cloud drops and before any rerun:** the replay showed that only 2 of the 20 number drops of `full_cloud` (E2-20261104T011906729928Z) are word-against-digit cases; the new value rule alone fixes only those. 16 of the 20 say "the two statements", "the two" or "different people" about their own evidence: the "two" counts the cited turns, and no quote contains it. The rule therefore also supports a spelled-out number word from two up to the number of distinct turns the finding cites, because that number counts the finding's own evidence. Digits are not covered (`2` still has to be in a quote), and a spelled number larger than the count of cited turns still drops. Cost: a finding that cites two turns may say "two" for something else without being dropped, which the judge, not this rule, would have to catch; this is the rule's known false-keep.
- **Amendment 2, written after checking the new rule against the 3,311 findings the saved E2 and E3 cards kept, and before any rerun:** 42 of them would now drop, mostly because the spelled words `first`, `second` and `third` are everyday adverbs and nouns ("she first said", "the second statement"), not numbers. In a statement these three words alone are no longer read as numbers (as before the fix); written as digits (`1st`) or inside a compound (`twenty-first`) they are, and in a quote they still count, so `1st` is supported by `first`. `fourth` and beyond are still checked. Four findings (two cards, each in two records) still failed because `first one` parsed as two numbers; a loose word beside a pronoun `one` is now dropped from the run, and none of the 3,311 fails. Their limit is added to the named limits: `the second floor` against a quote without it passes.
- **Frozen (coordinator, 2026-11-04, written before the rerun of `full_cloud`):** the rule is fixed as of commit `c1e2fc5`, including the allowance for a spelled number up to the count of cited turns. Reason: the old rule withheld true findings from the cloud arm, which understates it in a comparison the report makes, and the allowance is the one change that fixes most of those drops. It is not changed after seeing rerun results, and it applies the same way to every system and to T9.1. The rerun report gives how many findings were kept only because of the allowance and how many of those the judge scored unsupported, so the false-keep is measured rather than assumed narrow; a high count goes to the coordinator and nothing is adjusted. The old `full_cloud` record stays, listed in `supersedes`, and old and new numbers are reported side by side. E3 `single`'s one surviving drop is named in a record, with whether any E3 metric changes; a changed number is regenerated from the rows.
- **Would change it if:** the coordinator decides so on the measured false-keep count. Seeing that the rule hurts or helps a system is not by itself a reason.

## 42. What the corrected number rule would change in E2 and E3, from saved `drops` (no model call)
- **Date / task:** 2026-11-04, T7.1
- **Who:** The implementer, for the coordinator.
- **Method:** `contextlens.eval.number_replay` applies the rule of entry 41 to each number drop saved in a record's rows, against the full text of the turns the drop cited. A quote is part of its turn, so a survivor count is an upper bound. Rows that predate `drops` show none (`drops_saved_in_rows` false), so the two oldest E2 records cannot be replayed. The counts are reproduced by `tests/test_number_replay.py`.
- **Result, effective records:** E2-20261104T011906729928Z `full_cloud`: 20 number drops, 16 would survive, and 15 answerable cases that ended abstained have a surviving drop and no other drop behind them. E2-20261104T011721815005Z (`recent`, `full`, `retrieval`, `retrieval_memory`): 1 number drop in `retrieval_memory`, 0 survive; the others have none. E3-20261104T011047113852Z: `single` 1 drop, 1 survives (its card was not abstained); `retrieval_memory` 1 drop, 0 survive; `recent` and `team` none. The superseded records give the same counts for the same systems.
- **What the replay found, in words:** the cloud arm is the system that changes behaviour. Only 2 of its 20 drops are a number word against a digit; 16 are statements that say "the two statements" or "the two" about the turns they cite, which the amendment in entry 41 covers. Four still drop: two repeat or name a number their one cited turn does not hold (an injected "ten kilos", a "5,000"), and two say "the two" while citing one turn. Local arms and the team are essentially unaffected, so E3's comparison between `single` and `team` does not move, and E2's local ranking does not move. Only the cloud arm's answer rate would rise, and `full_cloud` would need a rerun for that to be a record.
- **Not done:** no rerun. Records that a rerun would affect are `E2-20261104T011906729928Z` (its supersedes entry would name this entry); nothing else changes behaviour by this count.

## 43. E6: what it compares, how a version is chosen, and the system it runs on
- **Date / task:** 2026-11-04, T7.1 (the selection rule was committed before the runner, 8ca44d2; the system setting follows the coordinator's decision of the same day)
- **Who:** The implementer; the coordinator chose the system.
- **Chosen:**
  - **Versions:** v1 is the baseline (E2's prompt); v2 to v7 each change one factor: few-shot examples, evidence format, abstention instructions, output format, length limits, role split. Each declares its added and removed lines in `factor.json`, and a test fails if the declared change is not the real line difference from v1. A prompt that defines the card modes is not a factor: T4.2 sets respond in code, and T3.3's mode-defining prompt stays reverted. Respond mode and injected claims, in findings and in suggestions, are measured for every version and are not in the score.
  - **Selection rule (`SELECTION_RULE`, in the record):** the score is the mean of `grounded_rate` and the lower of `abstain_when_insufficient` and `answer_when_answerable`. A candidate qualifies only with a 95% interval above zero on `grounded_rate` or `abstain_when_insufficient`, no interval below zero on any of the three, and a score above v1's; otherwise v1 is kept and the record says there was no measurable improvement. Six candidates are compared with no multiple-comparison correction, so a qualifier is a candidate for a confirming run.
  - **System:** a named setting, `--system single|retrieval_memory`, default `single` (E3's written rule selects it). The record names it in `config.system`. `retrieval_memory`'s weak spots seen in E2 stay an observation, not E6's target. If the team attempt later changes the selection under the written rule, only E6's chosen prompt and v1 are rerun on the new system, not all versions. `single` builds its system message from the version under test (checked by a test on every version), and the record also stores a hash of the system message the agent code builds.
  - **Cost, by counting calls, for 7 versions on 148 cases:** both systems make 1,036 card steps at least and build memory notes with 148 extraction calls, 3 canary calls and 2 pings; the judge adds one call per finding (about 890 at the finding counts of the current E3 record). Worst case for the writer is 4,144 calls on `retrieval_memory` and 10,360 on `single` (each of up to 4 steps and its retry, then the suggestion and its retry). In the current E3 record the two have the same median card time (about 2.8 s) and about one call per card, so each run is about an hour of writer time plus judging.
  - **Runs:** cases run one at a time, since one local model serves them and the 8 s deadline would be distorted by concurrency. Rows are saved as they arrive and reused only at the same commit. A version whose output is invalid or too slow to measure is listed in `attempts` and cannot qualify; failed calls stop the run.
- **Considered:** running all versions on both systems (twice the cost, and the second system's result has no use unless selection moves); `retrieval_memory` as the default (it was chosen on E2 numbers the number rule and short-id bug had affected).
- **Would change it if:** the team attempt changes E3's selection, or the single agent's prompt turns out to have a part that is not the version text (the hash of the full system message would show it).
