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
- Choose settings on **dev**. Use **test** once, on the chosen configuration. If test results influence a decision, record that and draw a new split.
- Keep public benchmarks and our own scenarios separate in every report. Label synthetic data as synthetic.

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
| **Cost** | Tokens per card, and dollars for cloud models |
| **Calibration** | Expected calibration error, for models that output a probability |

**Judging "supports the claim":** entailment judged by a rubric prompt on the strongest local model, plus person review of a random sample (H2 format). Report the agreement rate. Don't use the same model and prompt to both write and grade without saying so.

## Comparing systems
Run every system on the same cases with the same seed. Report a 95% bootstrap confidence interval (10,000 resamples at the conversation level) and an exact McNemar test for paired correct/incorrect outcomes. Don't call anything an improvement if the interval for the difference includes zero. Say "no measurable difference".

## Run records
Every experiment writes `results/runs/<experiment>-<timestamp>.json`:
`run_id, experiment, git_commit, dataset_hash, split, config, models, prompt_versions, n, metrics, intervals, attempts, spend_usd, predictions_path, started_at, finished_at`.
Predictions are saved, so metrics can be recomputed. `attempts` lists every setting tried, including the failures.

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
