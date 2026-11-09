# ContextLens report

Every number in this report comes from a run record in `results/runs/`, entered through `results/claims.json`. Where a figure is not yet allowed by that route, the text marks it pending and does not guess.
Run ids are written in full so each statement can be traced. Entry numbers point to `docs/DECISIONS.md`.

## 1. The problem and the question

During a conversation (a meeting, a class, a long discussion) a person loses track of what was said. ContextLens answers one question at the press of a button:
what was actually said, and is there enough to say it? The answer is one short card whose every statement cites the turn it came from, and "not enough information" is a
valid card. The design question (`docs/DESIGN.md`) is how an assistant can find the right context quickly enough to help, without inventing memories or mistaking a legitimate
update for a contradiction.

The project was also built to be measured: a team of agents against one agent with the same tools, prompting against fine-tuning, and a streaming speech stack.
The result that matters most is not the hoped-for one: the multi-agent team showed no clear quality gain over the single agent on the held-out test and costs far more (section 5, E8).

## 2. Architecture

A press freezes the conversation at one point and builds a card from the turns before it.

- **Speech:** a C++ service wraps whisper.cpp behind gRPC; speaker labels come from local diarization. A voice gate in the service drops silence and, as E7 shows, quiet speech.
- **Backend:** FastAPI, a transcript store (in memory or PostgreSQL with pgvector), a hybrid BM25 and embedding index over one-turn chunks, and a memory extractor that records decisions, commitments, questions and corrections with links to what they replace.
- **Agents:** LangGraph. A memory agent (search, surrounding turns, memory lookup), a consistency agent (search, consistency check) and an assistance agent that writes the card and may send one specialist back once. A controller enforces an 8 s deadline and returns what it has as a partial card. The single agent has the same tools and no checker.
- **Evidence guard, in code and not in the prompt:** after the model writes, a guard drops any finding that cites a turn that does not exist, a turn later than the press, or a quote that is not in the turn, and drops numbers that no cited quote holds. If nothing is left the card abstains. A prompt that asked the model to be honest about this was tried and reverted (entry 14).
- **Instruction guard:** word patterns remove turns that give orders to the assistant before a checker or the card writer reads them. It is a limit, not a protection (section 8).
- **Clients and observability:** a React and TypeScript client and a Unity client, OpenTelemetry traces and Prometheus metrics with no transcript text in them.

The product setting `CONTEXTLENS_ASSISTANT` picks `team`, `single` or `team_shipping_changed`. The default stays `team`; which system ships is the person's decision (section 7).

## 3. Data

All data is public or invented; nothing is a real person's conversation. Sources, licences and what was read are in `docs/datasets.md`; raw text is never committed.

- **Our scenarios:** written conversations with a stated label per case: answerable recall, insufficient evidence, contradiction, a legitimate update, a joke or hypothetical, a prompt injection, a wrong-speaker trap. The labels are reviewed. The dev split is the 148 cases that choices were made on.
- **New cases (`scenarios_extra`):** dev cases built to give the abstention comparison more insufficient-evidence cases. They are made from invented templates by a seeded generator, as pairs: an insufficient case and an answerable twin. **The labels are template-built and were read by an agent, not a person.** The run records mark them unreviewed, so `claims check` refuses any figure from them and none appears here. A template can be learned, and for some shapes a plain word-overlap threshold separates the pair. These cases guided development only.
- **DECODE** (contradiction detection) and **QMSum** (meetings), whose published labels are used; **LongMemEval** (a small search-only slice); **AMI** meeting audio and annotations for the speech stack, from the Hugging Face copy `edinburghcstr/ami`, rebuilt into one WAV per meeting from utterances placed at their timestamps (entry 50). Silence between utterances may flatter timing and speaker error.
- **Held-out test:** 136 card cases from our scenarios plus search-only cases from QMSum and LongMemEval, frozen before the test run (`configs/e8_freeze.json`). The test was run once.

## 4. How things were measured

- **Metrics.** `abstain_when_insufficient` is the share of insufficient-evidence cases where the card abstained. `answer_when_answerable` is the share of answerable cases where the card did not abstain. It counts cards that answered, not cards that were right (entry 38); do not read it as accuracy.
- **Differences** are called better or worse only when the 95% bootstrap interval excludes zero and, for the paired yes-or-no metrics, the exact McNemar test agrees. Otherwise the finding is "no clear difference".
- **Selection rules were written before the runs** (entries 10, 30, 43, 48, 51, 60, 64) and each result was applied as written, including the ones that went against the builder.
- **The judge.** Groundedness and unsupported-claim rates need a judge of whether a cited quote supports a statement. A local judge and then Haiku (through `claude -p`) graded claims; Sonnet graded a seeded sample. **Haiku failed its written calibration test against Sonnet (entry 56, record JCAL-20261105T044839474906Z) and is still the judge, by the coordinator's decision, because local judging was ruled out. No person reviewed any judge.** `claims check` therefore refuses every grounded-rate and unsupported-claim figure, and none is stated here. In E8's Sonnet sample (in the record), Haiku said "supported" where Sonnet said not far more often than the reverse, so Haiku leans lenient. Sonnet is a reference, not ground truth.
- **Where both judges wobble:** statements that restate a conflict ("A says X, B says not X"). Whether that restatement is supported is arguable, and the two judges split on it.

## 5. Experiments, in order, with the failures

### E1. Retrieval (record E1-20261103T035342386305Z, 414 dev queries)
Eight settings: dense, BM25, hybrid and hybrid with a reranker, each over one-turn chunks and 3-turn windows. By the rule written first (highest mean recall at 5, ties to MRR), hybrid over one-turn chunks won with recall at 5 of 0.768, against 0.732 for dense, 0.681 for BM25 and 0.754 with the reranker. The lead over dense and over the reranker is close to a tie, so **no claim is made that reranking helps**.

### E2. What context to give a single model (records E2-20261104T011721815005Z and E2-20261104T011906729928Z, 148 dev cases)
Four context shapes with the local 4B writer: recent turns, the full conversation, retrieval, and retrieval with memory. The three simple shapes abstained when they should (0.938) but answered only about two thirds of answerable cases (0.659 for recent, 0.674 for full and retrieval). Adding memory raised answering to 0.947 and cut correct abstention to 0.5. A cloud arm (Sonnet) abstained on every insufficient case (1.0) and answered 0.871.

Failures, stated plainly:
- A **short-id citation bug** (models cite `t0`, the guard knew only the full id) dropped true findings, most from the team. E2 and E3 were rerun after the fix (entry 36).
- A **number rule that compared spelled numbers as text** dropped true findings, most of them from the cloud arm. The rule was rewritten to compare values (entry 41). **The cloud arm's E2 rerun under the fixed rule was cancelled, so the E2 cloud numbers above carry the old rule's bug** and understate the cloud arm. The cloud model is not used as ground truth for any claim.
- A **mode-defining prompt** was tried to make "respond" cards appear, did not, answered fewer answerable cases, and let an injected claim through. It was reverted (entry 14).

### E3. Single agent against a team (record E3-20261104T011047113852Z, 148 dev cases)
Single and team have the same tools, no consistency checker, and the same 8 s deadline. The single agent abstained correctly on 0.875 of insufficient cases and answered 0.947; the team abstained correctly on only 0.312 and answered 0.977, at 1535 tokens per card against 634. A read of the saved rows (entry 48) found why: the memory specialist correctly found nothing, but the controller marked the card answered whenever the consistency specialist wrote any finding, even "the transcript does not give this detail", so memory's "not enough evidence" signal was outvoted.

### E3 team attempts (records E3T-20261104T131120203662Z, E3TS-20261104T173517541324Z)
The coordinator allowed one fix (`memory_gap_abstains`: abstain when memory finished cleanly with nothing) and, once, the combination that could actually ship (the fix plus the product's consistency checker, called `team_shipping_changed`, entry 51). On the new template-built cases the shippable team abstained correctly more often than the single agent, with a lower grounded rate under the cloud grader; so its gain is in abstention, not in the accuracy of what it says. Because those labels are unreviewed, no figure is claimed. The old-case gain of the fix is in-sample: the hypothesis came from reading those rows. The rule in entry 51 made `team_shipping_changed` the primary system of the test.

### E4. Contradiction detection (records E4-20261103T102658868337Z and E4-20261103T102930546849Z)
A DeBERTa-v3 checker was fine-tuned on DECODE plus an invented train-only set and compared with a prompted checker, a zero-shot natural-language-inference checker and a TF-IDF baseline. The first run's training loss was NaN in every setting (float16 weights; AdamW epsilon underflowed); its records are void and the rerun loads float32 (entry 6). The trained model had the best accuracy and the prompted checker far fewer false alarms, so both were kept and combined into a cascade with a threshold chosen by a rule written first (entries 7 and 8). Benchmark figures for the trained checker against the prompted checker and TF-IDF: macro-F1 on our test scenarios 0.729 (prompted 0.575, TF-IDF 0.189); contradiction F1 on DECODE, on published labels, 0.913 (TF-IDF 0.753).

### E5. Fine-tuned cards against prompting (record E5-20261106T222951150994Z, 148 dev cases)
A QLoRA adapter was trained to write cards and compared with the prompted version of the same base on one runtime, with the winner rule written first (entry 60). **The base moved from a 4B model to a 2B model for GPU memory**, so this says nothing about a larger adapter. The adapter learned to answer nearly everything: 0.962 of answerable cases against 0.212 for prompting, but it abstained correctly on only 0.438 of insufficient cases against 0.938 (difference -0.5, interval -0.75 to -0.25). Both produced valid schema 1.0 of the time. Under the unreviewed judge the adapter was also less grounded, and read by hand its citations were real, so the loss is unsupported content and not broken plumbing. The adapter is slower (23.9 s median against 7.2 s). **Prompting stayed.** Caveat: the prompted 2B card answers only 0.212 of answerable cases, so neither arm is a good product at that size.

### E6. Prompt versions (records E6-20261105T223904724358Z on `single`, E6-20261106T000633991380Z on the shipping team, 148 dev cases)
Four versions were compared (v1 baseline, v3, v4, v6; the person cut three of the seven planned), each changing one factor, on the single agent.
- v1: abstained 0.812, answered 0.947.
- v3: abstained 0.812, answered 0.955, at 602 tokens per card against 634.
- v4 abstained on everything (1.0) and answered only 0.311: over-abstention.
- v6 answered 0.977 but abstained correctly on only 0.312: under-abstention.

v3 was selected by the rule written first. **Its gain is in groundedness under Haiku, and Sonnet, grading every claim of v1 and v3 on the same rows, did not confirm it as measurable**: it points the same way but the gain is not measurable, so the judges split on `single` (entry 58). On the shipping team both judges chose v3. No groundedness figure is stated here because no person reviewed a judge. These runs used the number-rule fix of entry 41 and ran without the later instruction guard.

### E7. Streaming speech and real-time cards (records E7-20261107T003136847403Z, E7-20261107T011123157183Z, E7-20261107T023701185115Z, clean cards E7-20261106T193023811323Z)
Whole AMI meetings were streamed through the speech service at four times real speed, with questions drawn from AMI annotations.
- **A measurement bug, found and fixed.** The streamed word error rate was far worse than offline Whisper on the same audio because the service's fixed voice gate dropped most quiet speech. I found it with an offline control, explained it from the gate's threshold and the audio level (entry 68), and gave each rebuilt meeting one gain before streaming. **That gain is a normalization of rebuilt test audio, computed over a whole meeting, which a live system cannot do. It is not a live feature.** The fixed loudness gate in the product still drops quiet microphones; an adaptive gate is in `docs/backlog.md`.
- After the gain: streamed word error rate 0.376 and speaker error 0.599, both on published labels (AMI's own transcript and speaker segments; record E7-20261107T003136847403Z) (the earlier records were invalid for speech and are superseded; their clean-transcript cards are still valid and cited by id).
- **Latency.** Most cards on live audio were cut by the 8 s deadline, so latency figures are ceilings, not typical response times: button-to-card latency on the clean-transcript cards (record E7-20261107T023701185115Z) has a median of 5953 ms and a 95th percentile of 8524.4 ms, which is above the deadline.
- **Two latency fixes, both dropped.** Retrieval before the memory model call, and a shorter sub-deadline for the specialists, were given a keep-or-drop rule before the controlled before-and-after run (entry 64). On the same segment and the same 33 clean questions, the answer rate fell from 0.848 to 0.758, recall at 5 fell from 0.818 to 0.697, tokens per card rose from 1363 to 1676, and more cards were cut by the deadline. Both fixes were dropped (entry 70). That is a three-question drop on a small sample, not an effect size: the clean system's answer rate also moved between its earlier run and the rerun with nothing changed. No latency or token cut is claimed: the two optimizations tried made things worse.

### E8. The held-out test (record E8-20261107T073426291017Z, which supersedes E8-20261107T072458163837Z with the same rows; detectors and search in E8X-20261107T072457980506Z)
Run once, on 136 test cases, after configurations were chosen on dev. Two card systems were compared, the shipping team (`team_shipping_changed`, primary) and `single`, with prompt v3 and the instruction guard on. The trained-team arm (the adapter in the team) was **dropped from E8 by the person** (entry 75): the adapter had lost on dev and the product will not use it.

| Measure | single | shipping team | difference (single minus team) and 95% interval |
|---|---|---|---|
| tokens per card | 555 | 1349 | -795 (-820 to -769) |
| median latency, ms | 2832 | 4814 | -1983 (-2229 to -1623) |
| 95th percentile latency, ms | 4859 | 6829 | not compared |
| abstained when evidence was insufficient | 0.750 | 0.812 | -0.062 (-0.211 to 0.000), McNemar p 1.0 |
| answered when answerable | 1.000 | 0.992 | 0.008 (interval in the record) |
| speaker accuracy | 1.000 | 1.000 | none |

**Result:** the shipping team against `single` shows **no clear difference on any quality metric** and is **clearly worse on tokens per card and on median latency**. Groundedness and unsupported-claim rates are in the record and compare as no clear difference, but are not stated here for the judge reason in section 4. Sonnet's sample of graded claims is in the record. Search-only benchmark: hybrid retrieval on QMSum, on published labels, has MRR 0.530 and recall at 5 of 0.700 over 367 questions; the DeBERTa detector reaches macro-F1 0.729 against 0.575 (prompted) and 0.189 (TF-IDF).

## 6. What the team result means

The design said that if the single agent wins, that is reported and the architecture bullet changes. The team's gain, so far, is in abstention (on dev, on template-built cases), not in evidence: under Sonnet the team's statements are not better supported, and on the held-out test the abstention difference is not clear. So no claim is made that the multi-agent design improves grounded answers. The single agent is the efficient choice, at well under half the tokens and about 0.6 of the median latency, with no quality loss we can state.

## 7. Selected system against shipping system

Several systems are named in this project, and no sentence here or in the bullets describes one system's numbers as another's.
- **Selected on dev:** `single`, declared in entry 49 and the one the product would use if efficiency decides.
- **Measured team, and primary on the test:** `team_shipping_changed`, the team as the product builds it (with its consistency checker) plus the memory-gap rule, a one-time combination (entry 51). On dev its written score beat `single` on the new template-built cases under both judges, which made it the primary of E8.
- **The product default** is still `team`, which has neither the memory-gap rule nor the settings the test measured. `CONTEXTLENS_ASSISTANT` selects the assistant; which one ships is the person's decision and has not been made.
- **Gaps:** E3's team numbers describe `team` (no checker, no memory-gap rule); the E8 numbers describe `team_shipping_changed`; the system the dev rule ranked highest, `team_changed`, has no checker and cannot ship as measured.

## 8. Limits

- **No judge was reviewed by a person.** Haiku failed its written calibration, is the judge by decision, and leans lenient against Sonnet. Groundedness figures exist in the records and are deliberately not in this report.
- **Statements that restate a conflict are where both judges wobble.** Whether the restatement is supported is arguable, so these are the least reliable part of any judged rate.
- **New-case labels are template-built and agent-read.** They guided development; a template can be learned, and a plain overlap threshold can separate some pairs.
- **The number rule has a known false keep (entry 41).** A finding that cites N turns may say a spelled number up to N ("the two companies") with no quote holding it, so an invented count over two cited turns survives for a judge to catch. Ordinary `first`, `second` and `third` in a statement are not checked at all. The rule was frozen before the cloud arm's rerun so the result could not tune it, and **that rerun was cancelled**, so no run measured how many findings were kept only by this allowance, and no count is given.
- **The rule-based injection guard is weak and nothing claims protection.** Tested once on a blind writer's set of orders and ordinary turns, it caught only a minority of the orders and dropped none of the ordinary turns: it caught 14 of 60 orders and dropped 0 of 60 ordinary turns (results/guard-heldout/06440351937c.json). A trained classifier was started and cancelled by the person.
- **Audio.** The per-meeting gain is a normalization of rebuilt audio, not a live feature; the fixed loudness gate drops quiet microphones; most cards on live audio were cut by the 8 s deadline; meetings are rebuilt from utterances, so silence between them may flatter timing and diarization.
- **E5 is a 2B base** and says nothing about a larger adapter.
- **E2's cloud arm carries the old number rule's bug.**
- **Small samples.** E7's before-and-after comparison has 33 clean questions, so one question moves its answer rate by several points.
- **Held-out use.** The test split was run once, after a configuration was chosen on dev.

## 9. Reproducing and demonstrating

Setup, tests and the demo script are in `README.md`. To recheck the numbers here:

```
python -m contextlens.eval.claims check
python tools/check_report_numbers.py
```

The second fails while any pending placeholder above remains, which is expected until the open decisions are settled.
