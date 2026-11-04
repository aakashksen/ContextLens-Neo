# Extra dev cases (`dev_extra`)

108 synthetic cases: 54 pairs, each an `insufficient_evidence` case and an answerable twin (category `recall`).
Dev only. They are a separate scenario source, `scenarios_extra`, in `datasets/scenarios_extra/dev.jsonl` with the
manifest `datasets/splits/scenarios-extra.json`. `datasets/scenarios/dev.jsonl`, `test.jsonl` and
`datasets/splits/scenarios.json` are unchanged (a test pins their hashes). Ids look like
`syn-xtra-<shape>-<k>-ins` and `...-ans`; they do not collide with any dev or test id.

## How they were made
`src/contextlens/data/extra_cases.py` generates them with a fixed seed (`20261103`): 30 invented topic tuples
(thing, other thing, asked detail, value, another detail, its value), nine shape templates, 24 invented filler
lines, and 12 invented first names. Run `python -m contextlens.data.extra_cases` to rewrite the file, the manifest
and the review sample; a test checks the committed file is exactly the generator's output. Nothing is taken from
QMSum, DECODE, LongMemEval or any other dataset, and no conversation text of dev or test is reused (a test checks
that no turn of five or more words repeats and that no five-word run is shared).

**The author saw no results.** The templates and topics were written before looking at any run record,
prediction, error analysis or metric, and no system was run on these cases, so none of their wording was tuned
to what any system gets wrong.

## The rule for a pair
The twin has the same topic, shape, question, speakers and turn count as the insufficient case. They differ in
one turn only (in `after-cut`, the same two turns swap order), and that difference is whether the answer is in the
conversation up to `as_of_seq`. Twins carry `gold_evidence_turn_ids` (one turn) and a `gold_answer`; insufficient
cases have `gold_label` `insufficient`, no evidence and no answer. For half the shapes the insufficient case still
contains the asked value, as a guess, hypothetical or detail of another thing; that is the point of the shape.
Pairs with a long conversation (every third pair of a shape, 18 pairs) have 12 to 14 unrelated filler turns first
(15 to 17 turns); the others have 3 turns.

| shape | pairs | what the insufficient case has |
|---|---|---|
| topic-no-detail | 6 | the topic is mentioned, the asked detail never given |
| other-entity | 6 | the asked detail, but for another thing |
| after-cut | 6 | the detail only after `as_of_seq` (the one shape where `as_of_seq` is not the last turn) |
| promise | 6 | someone promises to look it up and report back |
| hypothetical | 6 | the value only inside an if-we-did-it sentence |
| speculation | 6 | the value as a guess nobody confirmed |
| never-mentioned | 6 | the thing asked about is never mentioned |
| explicit-unknown | 6 | a speaker says they have not heard the detail yet |
| different-detail | 6 | another detail of the same thing, not the asked one |

## Labels
`label_source` is `unreviewed_extra`: no person has reviewed these labels, so `labels_reviewed` is false for
this source whatever H2's state, and any claim built on them must say so. `check_run` validates a record of this
source against its manifest like the other local sources. `datasets/scenarios/dev-extra-review.md` holds 15 pairs
(one per shape, then random) for a reader.

## Known weaknesses (checked by reading the sample and by scripted searches)
1. **Answerable by inference or arguable labels.** Read all 15 sampled pairs, and searched all 54 for the asked
   value in the insufficient case (outside the shapes where it is meant to appear) and for asked-detail or value
   words in filler turns (none found). The `different-detail` details were read by eye: none implies the asked
   one. The label is arguable for `hypothetical` and `speculation`: a lenient reader may take a stated guess as an
   answer. They are labelled insufficient because the conversation never asserts the value.
2. **The twin's evidence is too obvious.** The evidence turn repeats the question's wording (the question's
   content words are 83% covered on average, at least 33%), so the twins are easy recall cases. A score computed
   only on the twins says little; the pairs are meant to be compared.
3. **Template regularities and lexical cues.** The wording is templated, so a system can learn the phrases. A
   scripted check of question-word overlap with the conversation up to `as_of_seq` shows that in
   `topic-no-detail`, `after-cut`, `never-mentioned` and `different-detail` the insufficient case always has
   lower overlap than its twin, so a plain retrieval-score threshold can separate those 24 pairs without reading
   the meaning. In the other five shapes the overlap is equal. Treat those four shapes as the weaker test.
