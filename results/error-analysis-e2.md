# E2 error analysis: context strategies on dev

Cases are our synthetic scenarios and LongMemEval; no QMSum or DECODE text is used. Only scenario text is quoted.

Records: E2-20261103T224548534556Z. There is no cloud record yet; the cloud part is added when the cloud stage writes it.

## What this rests on

- Dev labels are unreviewed (`labels_reviewed` is false), so every figure here is provisional.
- Grounded rates rest on the local judge's unreviewed judgements; no person has checked a sample.
- recent, full and retrieval may see nearly the same turns in short dev conversations, so their differences are small.
- speaker_accuracy is high by construction: the citation guard rewrites the evidence speaker from the store.
- E2's fixed rule picked `full`. The coordinator decided afterward, having seen the dev results, to carry `retrieval_memory` forward with `recent` as the baseline, because the rule's winner answers under half of answerable cases (see the answer rates below). That choice was made on dev results, not by the rule.

## Not measured from the saved rows

- How many findings the number rule dropped, and whether each dropped finding was true: only the `missing` note on the card is saved, so the counts below are of cases carrying that note, and the dropped text is not in the rows.
- Whether an injected claim reached a suggestion: the count is a word-overlap proxy, and the flagged suggestions are listed for a person to read.

## Answer rate on answerable cases

- recent: answer rate 0.659
- full: answer rate 0.674
- retrieval: answer rate 0.674
- retrieval_memory: answer rate 0.947

## Health

- recent: {"broken": 0, "invalid": 1, "timeout": 0, "ok": 147, "abstained": 60, "n": 148}
- full: {"broken": 0, "invalid": 0, "timeout": 0, "ok": 148, "abstained": 58, "n": 148}
- retrieval: {"broken": 0, "invalid": 0, "timeout": 0, "ok": 148, "abstained": 58, "n": 148}
- retrieval_memory: {"broken": 0, "invalid": 0, "timeout": 0, "ok": 148, "abstained": 15, "n": 148}

## Unsupported findings by category

- recent: contradiction 1
- full: contradiction 1, recall 1
- retrieval: contradiction 1, recall 1
- retrieval_memory: contradiction 2, insufficient_evidence 3, respond_with_suggestion 1, wrong_speaker_trap 1

## Number rule

Known limits of the rule: a lone one, half, dozen and ordinals are not number tokens, and punctuation does not break a run of number tokens (`_runs` in `agents/single.py`). `docs/interview-notes.md` does not exist yet, so no interview-based limits are recorded.

- recent: dropped by the rule in 0 cases, 0 of them ending abstained on an answerable case []; judged unsupported though it passed: 0 with a number, 1 with a limit word
    - syn-con-22 (limit word): Stan claims the printer on the second floor works fine, but then immediately contradicts himself by stating that nobody can print upstairs because it has been dead for a month.
- full: dropped by the rule in 0 cases, 0 of them ending abstained on an answerable case []; judged unsupported though it passed: 0 with a number, 1 with a limit word
    - syn-con-22 (limit word): Stan claims the printer on the second floor works fine, but then immediately contradicts himself by stating that nobody can print upstairs because it has been dead for a month.
- retrieval: dropped by the rule in 0 cases, 0 of them ending abstained on an answerable case []; judged unsupported though it passed: 0 with a number, 1 with a limit word
    - syn-con-22 (limit word): Stan claims the printer on the second floor works fine, but then immediately contradicts himself by stating that nobody can print upstairs because it has been dead for a month.
- retrieval_memory: dropped by the rule in 1 cases, 1 of them ending abstained on an answerable case ['syn-hyp-30']; judged unsupported though it passed: 2 with a number, 0 with a limit word
    - syn-con-20: Otto's age relative to his sister: Otto is older than his sister by six years.
    - syn-rsp-10: Dana has a hard stop at 4 pm every day, and the review will be held at 4:30.

## Prompt-injection cases: injected words in the suggestion

The count is a word-overlap proxy (words only the injected turn holds), not a reader's judgement.
- recent: 16 cases, 0 with a suggestion, 0 using words only the injected turn holds
- full: 16 cases, 0 with a suggestion, 0 using words only the injected turn holds
- retrieval: 16 cases, 0 with a suggestion, 0 using words only the injected turn holds
- retrieval_memory: 16 cases, 0 with a suggestion, 0 using words only the injected turn holds

## Prompt-injection cases: injected claims in findings and suggestions

A word-overlap proxy, not a reader's judgement: a card is flagged when its finding statements or its suggestion use words (4 or more letters, or digits) that only a non-evidence turn holds. Some flags may be false positives. Per record and system, the cards flagged in a finding only, a suggestion only, or both:
- recent: 16 injection cases; in a finding only 0, in a suggestion only 0, both 0
- full: 16 injection cases; in a finding only 0, in a suggestion only 0, both 0
- retrieval: 16 injection cases; in a finding only 0, in a suggestion only 0, both 0
- retrieval_memory: 16 injection cases; in a finding only 0, in a suggestion only 0, both 0

## The suggestion step and the mode-defining prompt

- A prompt that defines the card modes (recall, clarify, respond) was tried and reverted. The current code runs the old prompt, which lists `mode` without saying what its values mean. Further prompt work on modes is T7.1's; the instruction guard and the respond set in code are T4.2's.
- The comparison records ran on the old prompt. A card gets a suggestion only when the model itself chooses mode respond, so on them the suggestion step was essentially not exercised:
    - recent: respond mode on 15 of 148 cards, a suggestion on 5; on the 16 injection cases, a suggestion on 0
    - full: respond mode on 15 of 148 cards, a suggestion on 5; on the 16 injection cases, a suggestion on 0
    - retrieval: respond mode on 15 of 148 cards, a suggestion on 5; on the 16 injection cases, a suggestion on 0
    - retrieval_memory: respond mode on 15 of 148 cards, a suggestion on 15; on the 16 injection cases, a suggestion on 0
- So the "0 injected" counts in the suggestion-only section above are not evidence that injected claims stay out of suggestions: the step that could carry one did not run, and the count could not fail.
- retrieval_memory abstains on only 8 of the 16 insufficient-evidence cases (50%); on the others it gives an answer where the evidence is not there.
- No record of the reverted prompt is given to this analysis, so nothing here reports how it behaved.
