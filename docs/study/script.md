# Facilitator script (about 30 minutes per person)

This is part of the human gate H4. Run 5 to 8 sessions, one person at a time. This counts as informal product
testing. If the results are to be published as research, check with UCSB's IRB first. Participants may include
neurodivergent people if they wish to join; do not ask anyone about this.

## Before the sessions
1. Review `consent.md` and fill in the bracketed parts. It is a draft and not approved by anyone.
2. Run `python -m contextlens.experiments.e9_study plan N` (N from 5 to 8) and keep the printout. Hand out
   codes P01, P02, ... in that order. Do not write a name next to a code anywhere.
3. Run the assistant on the local machine with the local model only. Check that no cloud model is configured, since
   the consent form says nothing is sent out.
4. Create `results/study/results.csv` by copying `docs/study/results-template.csv`. The `results/study/` folder is
   ignored by git so it is not committed by accident.

## For each person
1. Give them the consent form and time to read it. Answer questions. They sign on paper. File the page apart from
   the results.
2. Give them a slip with their code. Explain it is the only way to withdraw their results.
3. Explain the tasks: four invented conversations, each followed by one question; two with the assistant, two
   without; they may stop or skip at any time.
4. Do the four tasks in the order on their plan. For an assisted task, play the conversation into the assistant,
   and let the person ask it the question. For an alone task, give them the conversation on paper.
5. Score each answer with `answer-key.md` right after, and add one row per task to the results file:
   `participant_code,task_id,condition,points`.
6. Thank them. Do not write down comments, names, or anything they say about themselves.

## If someone stops or withdraws
- Stops during the session: keep only the finished tasks if they agree, otherwise delete their rows. The analysis
  leaves out any kind of task where one of the pair is missing and counts the person only on what is complete.
- Withdraws later with their code: delete every row with that code from the results file. You can also pass the
  code with `--withdrawn` when running the analysis; the sample size shown then drops by one.

## After the sessions
`python -m contextlens.experiments.e9_study analyze results/study/results.csv`

It prints each person's difference, the mean with a 95% interval, and the method. It does not write to
`results/claims.json`. Anything you want to quote goes through the normal claims path.
Delete the results file and the paper forms on the date written in the consent form.
