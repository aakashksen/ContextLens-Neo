# E3: the 30 false alarms of the trained model alone, read by hand

What this is: the 30 false alarms drawn (seed 13) from the trained model alone at the selected threshold (the one in
`configs/cascade.json`), from the local file `results/local/false-alarms-trained.md`. That file holds dataset text and
stays local; **this file holds counts and descriptions only, no DECODE text and no case ids from DECODE.**

Who read it: the implementer, once, in one pass. This is a reading by one agent, not a person's review, and the
grouping below puts each case under the feature that stood out most, so the groups do not overlap and are not exact.

Of the 30: 26 are DECODE dev cases and 4 are our synthetic wrong-speaker traps. All 30 are labelled consistent, and
the model's probability of contradiction on every one is high (0.95 to 0.998), so a higher threshold would not remove them.

## DECODE cases (26)

- **7: reactions with no factual claim** (agreement, sympathy, "wow", "that's a good idea"). Nothing is stated that
  could contradict anything.
- **8: a compatible statement that contains a negation or a hedge** ("never", "not", "don't", "doubt", "wish ... not").
  The model appears to read the negation word as the contradiction.
- **5: the same speaker adding to what they said** (an elaboration or a preference that fits the earlier turns).
- **4: a question, or the answer to one.**
- **2: facts about two different people** (two different ages, a stated favourite against another speaker's "no
  favourite"). These differ but nothing is contradicted, because they are about different people.

## Synthetic wrong-speaker traps (4)

All four are the case the trap is built for: the statement contradicts what **another speaker** said earlier ("I voted yes"
by one person, "I voted no" by another). The conflict is real, but it is between two people, so it is a disagreement and
not a person contradicting themselves. This is the pattern the same-speaker rule in the E3 consistency records addresses.
The DECODE cases cannot show it, because DECODE speakers are only A and B by position.

## What the pattern says

- Most DECODE false alarms (19 of 26: the reactions, the negation words and the questions) are ordinary conversation where
  the model reacts to a negation word or to topic overlap, not to an actual conflict. That is a property of the trained
  model and of the training data, not something a threshold fixes.
- The synthetic false alarms are all conflicts between two speakers.
- The cascade raised too few false alarms at this threshold to draw 30 from, which is why these come from the trained
  model alone. How many the cascade keeps is in the E3 consistency records.
- Limits: 30 cases, one reader, dev data whose labels came from the dataset's publisher. The grouping is a description,
  not a measurement, and no figure in it should be quoted as a result.
