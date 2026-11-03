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
- **Chosen:** rerun E3's `cards` part, E2's four local strategies, and E2's cloud arm, because the citation guard dropped true findings that cited a short turn id such as `t0`. Same cases, same seeds, same config apart from the fix and an exclusive GPU. E2's cloud arm also moves to Sonnet, by the person's rule on Claude comparisons, on the same committed subset, and its record says so. The old records stay listed in `supersedes`, with this bug as the reason. Selection follows the written rule, whichever system it favours. The 8 s deadline doesn't change during these runs.
- **Considered:** keeping the old records with a caveat. It lost because the bug hits systems unequally, the team most, so the comparison itself can't be read. Fixing the bug and moving on to E6 lost too, since `retrieval_memory` was chosen on the affected numbers.
- **Evidence:** dropped citations with a short id, in E3-20261103T185316318532Z, counted from its saved rows, per `orchestration/evidence/cleanup-result-risks.md`.
- **Would change it:** nothing in the new numbers. If the rerun can't reproduce the same cases and seeds, that gets stated, not worked around.
