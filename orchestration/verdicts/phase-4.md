VERDICT: DEFECTS

Phase 4 review at build commit ec078ea. Scope: T4.1 (agent team), T4.2 (E3), records E3-20261104T011047113852Z (cards, dev, n=148), E3-20261104T013101638012Z (consistency, DECODE dev, n=1000) and E3-20261104T012006860469Z (consistency, scenarios), DECISIONS 28-39, and the two verdicts T4.1.json and T4.2.json.

What I did: read the plan, the design, both verdicts, graph.py and assistance_agent.py, and the E3 record fields (config, metrics, comparisons, notes). I ran one small Python check against `rebuild_records._rows`; I ran no test files, no check_run, and used no GPU, Ollama or network. I did not read any `datasets/**/test*` path. All E3 and JC records are `split: dev`, `dirty: false`; no test-split use found. `results/claims.json` is empty, and README and RESUME-TARGETS hold no E3 numbers (only `[grounded_gain]` placeholders).

What holds up:
- The finding is stated as it is. `notes[0]` of the cards record says the team is not clearly better than the single agent on anything and is worse on abstaining where evidence is insufficient (0.3125 vs 0.875, 0 vs 9 discordant cases of 16, exact McNemar p=0.004), tokens (1535 vs 634 per card) and p50 latency (3908 vs 2834 ms). It notes that `answer_when_answerable` counts non-abstentions, not correct answers, and that the interval and McNemar disagree on it.
- Drops by reason, wrongful-drop checks, duplicate findings (38 team repeats, with the grounded rate recomputed without them), the small public slice (10 of 148) and the vacuous suggestion-injection count (no injection case produced a suggestion) are all disclosed in the record.
- The `retrieval_memory`/`recent` choice after dev results is disclosed in `choice_note`.
- Flip rates above the 5% limit are flagged (`bring_to_coordinator_before_e8: true`): injection flips 21% and 13% for trained alone, 16% and 15% for the cascade.

## Defects that could put a false number on record

None found that is already on record as a claim. The claims file is empty. The two items below could mislead if their figures are copied into a claim, and both depend on a later task (E8) to handle correctly. I found no explicit plan line that owns D2.

### D1. The team that E3 measured is not the team the design describes or the product runs
- Files: `docs/DECISIONS.md` entry 34; `results/runs/E3-20261104T011047113852Z.json` `config.same_for_all` ("no consistency checker"); `src/contextlens/api/production.py` `_TeamPerPress.answer` (builds `PromptedChecker` and passes it to `Tools`).
- Evidence: DESIGN.md's agent table gives the Consistency agent the `check_consistency` tool. In E3 the team's consistency agent had only transcript and memory tools (`consistency_agent.system(prompt_version, has_checker=False)`). Production hands the team a `PromptedChecker`, and nothing wires `CascadeChecker` (built in T4.2) into the team anywhere.
- Impact: any "multi-agent vs single agent" figure from E3 describes a team without its consistency tool. The user-facing system is a different, unmeasured one. The resume bullet's `grounded_gain` (E8) must not be taken from E3 as if it covered the product. The record does disclose the missing checker, so this is not a hidden number. It is a mismatch between what was measured and what ships.

### D2. Calculated alert precision of 1.0 rests on zero observed false alerts
- File: `results/runs/E3-20261104T013101638012Z.json` `config.alert_precision`.
- Evidence: `trained_same_speaker` and `cascade_same_speaker` on scenarios show precision 1.0 at both 5% and 20%. The inputs are `far` 0/65 and `catch` 7/17 (`config.catch_far`). The formula is c*p/(c*p + f*(1-p)) with f=0 observed, so 1.0 is the arithmetic of a zero count, not a measured precision. The same figures use an operating point chosen on the same dev data (`selection_data: dev (selection data, not held out)`). `alert_precision_note` says "calculated, not measured" but does not say that a zero count was used or give an interval.
- Impact: if "100% precision" reaches a claim or the interview notes, it is a false number. Fix: report the upper bound on f (for 0/65, about 4.5% at 95%), or mark the figure as undefined. Same issue, smaller, for the DECODE cascade (far 1/503, precision 0.94 and 0.99).

## Lesser defects

### D3. The two label flags in one consistency record disagree
- File: `results/runs/E3-20261104T013101638012Z.json`. Top-level `labels_reviewed` is true (label source `publisher`; `label_note` says reviewed). `config.labels_reviewed` is false (reproduce: load the JSON and print both).
- Impact: `eval/claims.py` line 74 reads the top-level flag, so a claim would be gated on true, while a reader of the config sees false. `config.provenance.cascade_config_threshold_provisional` is also true. DECISIONS entry 8 and its caveat describe these dev labels as unreviewed. The record contradicts itself, and the stale value looks copied from the E4 record.

### D4. The pre-merge E3 cards record can no longer be rebuilt without a model call (confirmed, not fixed)
- Files: `src/contextlens/experiments/rebuild_records.py` lines 66-70 (`_rows`); `eval/records.py` (`Prediction` gained `cloud_supported`).
- Reproduce: `rebuild_records._rows(Path('.'), json.load(open('results/runs/E3-20261104T011047113852Z.json')))` raises `RunBroken: system recent: the saved rows do not write back to the same bytes`. T4.2's verifier named this (check 3c) and left it. Every current E3 and E2 record written before the field was added has the same problem.
- Impact: the "rebuild from saved rows" route, used three times in this phase to fix derived fields, is now closed for the records that matter. A later correction to McNemar or note wording would need a rerun, or a code change that writes `cloud_supported` only when set.

### D5. `hand_read_file` in the consistency record points to a deleted worktree
- File: `results/runs/E3-20261104T013101638012Z.json` `config.hand_read_file`: `<local path>`. That path does not exist (`os.path.exists` is False). A copy is at `results/local/false-alarms-trained.md` in the main folder (git-ignored).
- Impact: the record's evidence pointer for the required hand-read is dead. The text is DECODE-derived and rightly untracked, so the record should name the main-folder path.

### D6. A non-`LLMError` exception in the assistance step is not caught in the team
- Files: `src/contextlens/agents/graph.py` `_assist` (lines 140-175, no try block around `decide` or `_finish`); `src/contextlens/agents/assistance_agent.py` lines 66-68 catch `LLMError` only. T4.1's verifier noted the same: a `RuntimeError` raised inside the assistance agent propagates out of `team.answer`. I read the code and did not run it.
- Impact: the plan requires each agent to catch a store error and abstain, and the specialists do (`graph._specialist` catches `Exception`). The assistance step does not, so one unexpected exception during a press would escape `TeamAssistant.answer` instead of giving an abstained card.

### D7. The key abstention finding rests on 16 cases, while 54 extra dev cases were built and not used
- Evidence: the record says "rests on 16 insufficient-evidence cases, so one case moves it 6 points". `datasets/scenarios_extra/dev.jsonl` (54 insufficient/answerable pairs, commit 9d40ad9) is not used by any E3 record; the verdict says none exists yet.
- Impact: team-versus-single abstention is the result that most decides the architecture bullet. It stands on 9 discordant cases. This is disclosed and the p-value is exact, so it is not a false number. It is a missed chance to strengthen the one conclusion the design asks for. Also, the E3 team used the same deadline of 8 s and 4 of 148 team cards were cut by it (timeouts), so the team is slightly disadvantaged on a few cases; the record says to read this.

### D8. Injection-in-suggestion and respond-rule checks are weak, as disclosed, and the plan wording is only partly met
- Record: `config.injection_words_in_suggestion` shows 0 suggestions on 16 injection cases per system, so the zero cannot fail. `config.respond.rule_note` says the respond rule reads the latest turn rather than the latest other-speaker turn the plan names. The instruction guard's patterns were written after reading dev injection turns (`guard_note`), with 1 of 12 look-alikes falsely flagged and 1 known miss.
- Impact: none is hidden, and the blind guard set and its freeze are covered by `prep/injection`. The plan's "injected claims in suggestions, for both systems" is, in effect, unmeasured at the end of this phase.

## Answer to the closing question
No defect I found could put a false number on record with no later task owning it. D2 is the nearest: a calculated precision of 1.0 from a zero count. The plan does not name a task that will correct it, so it should be fixed in the record's note or in E8's reporting before any claim cites it. D1 is carried by E8 (the design's team has to be measured with its checker). D3, D4 and D5 are record-hygiene defects that make the records harder to trust or rebuild, not wrong figures.
