# Testing and evaluation

## Layers

| Layer | Runs | Needs | Who |
|---|---|---|---|
| **Unit and integration** | Every commit, `.venv/Scripts/python -m pytest -q` | Nothing: no GPU, network, or spending. Uses `FakeLLM` and fixtures | Implementer |
| **Acceptance** | `taskctl verify <id>` | Whatever the task lists | The gate |
| **Independent verification** | After each task | The task text and code, no build notes | `verifier` |
| **Phase review** | End of each phase | The whole phase together | `reviewer` |
| **Real runs** | `pytest -m real`, and experiments | Ollama, GPU, or Docker | Named tasks only |
| **CI** | Push, once the person publishes | Same as the first layer, plus prompt regression tests with recorded responses | GitHub Actions |

## Data splits
- Split by **conversation**, never by turn, so related turns can't leak across. A saved manifest (`datasets/splits/*.json`) with a content hash fixes each split.
- Choose settings on **dev**. Use **test** once, on the chosen configuration. One run record that holds every system is one use of test. If test results influence a decision, record that and draw a new split.
- Keep public benchmarks and our own scenarios separate in every report. Label synthetic data as synthetic.

## Provisional results
Dev experiments may run before the person reviews the labels (H2), with `labels_reviewed: false` in the record. They guide choices but aren't claims. The final test run (E8) needs H1 and H2. If H2 finds wrong labels, fix them and rerun what they affect.

## No peeking
Replay feeds turns in order and freezes `as_of_seq` at the button press. Every retrieval and tool call filters on it, and a test confirms that a
later turn can't appear in evidence.

## Metrics, each defined once in `contextlens/eval/metrics.py`

| Metric | Definition |
|---|---|
| **Recall@k, MRR** | Gold evidence turn found within the top k; mean reciprocal rank of the first gold turn |
| **Macro-F1** | Unweighted mean F1 over `consistent`, `update`, `contradiction`, `insufficient` |
| **False alert rate** | Share of non-contradiction cases labeled `contradiction` |
| **Grounded rate** | Share of card claims whose cited turn exists at or before `as_of_seq` and supports the claim (judged per the rubric) |
| **Unsupported claim rate** | Claims failing the above, per card |
| **Abstention accuracy** | Abstains when gold says insufficient, and doesn't when gold has an answer; report both sides |
| **Speaker accuracy** | Claims attributed to the correct speaker |
| **Latency** | Button to card p50 and p95, per stage |
| **Cost** | Tokens per card, and Claude Code calls used |
| **Calibration** | Expected calibration error, for models that output a probability |

**Judging "supports the claim":** entailment judged by a rubric prompt on `gemma4:e4b-it-qat`, a different local model from the writer, plus person review of a random sample (H2 format). Report the agreement rate. Don't use the same model and prompt to both write and grade without saying so. A record that saves `grounded_rate` or `unsupported_claim_rate` names the judge in `config.judge` (`model`, `rubric_version`, `same_model_and_prompt_as_writer`), and a claim on either also needs `config.judge_review` (`agreement_rate`, `sample_size`, `sample: "random"`). Code can check that these are recorded, not that the judgements are true.

## Comparing systems
Run every system on the same cases with the same seed, and save them in one run record. Report a 95% bootstrap confidence interval (10,000 resamples at the conversation level, seed 13) and an exact McNemar test for paired correct/incorrect outcomes. Each resample draws whole conversations and recomputes the actual metric on them with the functions in `metrics.py`, so macro-F1, rates, latency and the other metrics all get intervals and paired differences. If the metric is undefined on the data or on any resample, there is no interval or comparison for it (dropping those resamples would bias it). McNemar is only for correct/incorrect outcomes; a saved p-value on any other metric fails. `check_run` recomputes the saved intervals and comparisons from the predictions. Don't call anything an improvement if the interval for the difference includes zero. Say "no measurable difference".

## Run records
Every experiment writes `results/runs/<experiment>-<timestamp>.json`:
`run_id, experiment, git_commit, dataset_hash, split, config, models, prompt_versions, n, metrics, intervals, comparisons, attempts, claude_calls, labels_reviewed, predictions, started_at, finished_at`, and optionally `supersedes`.
One record holds every system of the experiment: `predictions` maps each system name to its predictions file, `metrics` and `intervals` map each system name to its values, and `comparisons` lists system pairs (`a`, `b`, `metric`, `diff`, `ci`, `mcnemar_p`, with `a` minus `b`). An experiment with one system has one entry. Every system covers the same scenario ids: the whole split, or a named subset. A record on a subset has a `subset` field naming `datasets/splits/subsets/<name>.json` (`split`, `ids`), which must be committed in a commit that is an ancestor of the record's `git_commit` and unchanged since; `check_run` requires exactly those ids once each, `n` equal to their count, and every id in the record's split. An unnamed subset, or one chosen after the run, fails.
Predictions are saved, so metrics can be recomputed. Rows are all cards or all retrieval rows, and cite only turns at or before `as_of_seq`. A metric whose source field is missing on any row can't be saved.
`check_run` checks every record of the experiment and split. A record is excluded only if a later one lists it in `supersedes` with a reason; the file stays. The test split has one record unless a later one has an `override_note`, whatever the experiment's name. `claims check` applies the same test-once rule to a claim on a test record. `attempts` lists every setting tried, including the failures.

## Experiments

| ID | Question | Task |
|---|---|---|
| E1 | Which retrieval setup finds evidence best: dense, BM25, hybrid, hybrid + rerank? | T2.5 |
| E2 | What does each context strategy add: recent turns, full transcript, retrieval, retrieval + memory? | T3.3 |
| E3 | Does a multi-agent team beat one agent on quality, latency, and cost? | T4.2 |
| E4 | Does a trained consistency model beat TF-IDF, zero-shot, and prompted baselines? | T5.2 |
| E5 | Does a fine-tuned small model write better cards than the same model prompted? | T6.2 |
| E6 | Which prompt choices actually move groundedness and abstention? | T7.1 |
| E7 | What do live speech and diarization errors cost compared to clean transcripts, and where does latency go? | T8.5 |
| E8 | Final comparison on the untouched test split | T9.1 |
| E9 | Do people get more correct, faster, with the assistant than without? | H4 |
