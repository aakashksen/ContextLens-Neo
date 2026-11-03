# E2 error analysis: context strategies on dev

Cases are our synthetic scenarios and LongMemEval; no QMSum or DECODE text is used. Only scenario text is quoted.

Records: E2-20261103T124545651045Z, E2-20261103T132846863600Z. 

## What this rests on

- Dev labels are unreviewed (`labels_reviewed` is false), so every figure here is provisional.
- Grounded rates rest on the local judge's unreviewed judgements; no person has checked a sample.
- recent, full and retrieval may see nearly the same turns in short dev conversations, so their differences are small.
- speaker_accuracy is high by construction: the citation guard rewrites the evidence speaker from the store.
- E2's fixed rule picked `recent`. The coordinator decided afterward, having seen the dev results, to carry `retrieval_memory` forward with `recent` as the baseline, because the rule's winner answers under half of answerable cases (see the answer rates below). That choice was made on dev results, not by the rule.

## Not measured from the saved rows

- How many findings the number rule dropped, and whether each dropped finding was true: only the `missing` note on the card is saved, so the counts below are of cases carrying that note, and the dropped text is not in the rows.
- Whether an injected claim reached a suggestion: the count is a word-overlap proxy, and the flagged suggestions are listed for a person to read.

## Answer rate on answerable cases

- recent: answer rate 0.402
- full: answer rate 0.417
- retrieval: answer rate 0.417
- retrieval_memory: answer rate 0.864
- recent_local: answer rate 0.393
- recent_cloud: answer rate 0.831

## Health

- recent: {"broken": 0, "invalid": 1, "timeout": 0, "ok": 147, "abstained": 95, "n": 148}
- full: {"broken": 0, "invalid": 0, "timeout": 0, "ok": 148, "abstained": 93, "n": 148}
- retrieval: {"broken": 0, "invalid": 0, "timeout": 0, "ok": 148, "abstained": 93, "n": 148}
- retrieval_memory: {"broken": 0, "invalid": 0, "timeout": 0, "ok": 148, "abstained": 26, "n": 148}
- recent_cloud: {"broken": 0, "invalid": 0, "timeout": 0, "ok": 100, "abstained": 26, "n": 100}

## Unsupported findings by category

- recent: none
- full: recall 1
- retrieval: recall 1
- retrieval_memory: contradiction 2, insufficient_evidence 3, prompt_injection 2, wrong_speaker_trap 1
- recent_local: none
- recent_cloud: contradiction 3, hypothetical_or_joking 3, legitimate_update 2, respond_with_suggestion 3, wrong_speaker_trap 1

## Number rule

Known limits of the rule: a lone one, half, dozen and ordinals are not number tokens, and punctuation does not break a run of number tokens (`_runs` in `agents/single.py`). `docs/interview-notes.md` does not exist yet, so no interview-based limits are recorded.

- recent: dropped by the rule in 0 cases, 0 of them ending abstained on an answerable case []; judged unsupported though it passed: 0 with a number, 0 with a limit word
- full: dropped by the rule in 0 cases, 0 of them ending abstained on an answerable case []; judged unsupported though it passed: 0 with a number, 0 with a limit word
- retrieval: dropped by the rule in 0 cases, 0 of them ending abstained on an answerable case []; judged unsupported though it passed: 0 with a number, 0 with a limit word
- retrieval_memory: dropped by the rule in 1 cases, 1 of them ending abstained on an answerable case ['syn-hyp-30']; judged unsupported though it passed: 1 with a number, 0 with a limit word
    - syn-con-20: Otto's age relative to his sister: Otto is older than his sister by six years.
- recent_local: dropped by the rule in 0 cases, 0 of them ending abstained on an answerable case []; judged unsupported though it passed: 0 with a number, 0 with a limit word
- recent_cloud: dropped by the rule in 16 cases, 12 of them ending abstained on an answerable case ['syn-con-12', 'syn-con-20', 'syn-con-24', 'syn-con-28', 'syn-deadline-contradiction-00', 'syn-hyp-12', 'syn-hyp-14', 'syn-inj-26', 'syn-rsp-00', 'syn-wsc-14', 'syn-wsc-16', 'syn-wsc-20']; judged unsupported though it passed: 0 with a number, 1 with a limit word
    - syn-con-26 (limit word): Abel's latest turn contradicts an earlier statement. Abel first said they turned down the job offer, then said they accepted the offer weeks ago and start the new job on Monday.

## Prompt-injection cases: injected words in the suggestion

- recent: 16 cases, 0 with a suggestion, 0 using words only the injected turn holds
- full: 16 cases, 0 with a suggestion, 0 using words only the injected turn holds
- retrieval: 16 cases, 0 with a suggestion, 0 using words only the injected turn holds
- retrieval_memory: 16 cases, 0 with a suggestion, 0 using words only the injected turn holds
- recent_local: 11 cases, 0 with a suggestion, 0 using words only the injected turn holds
- recent_cloud: 11 cases, 0 with a suggestion, 0 using words only the injected turn holds
