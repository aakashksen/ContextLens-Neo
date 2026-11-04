# Code map

Entry points were read from the code. Module paths are under `src/contextlens/` unless shown.

## Parts

- `agents/`: `graph.py` `TeamAssistant`. Memory and consistency agents run in parallel, then the assistance agent writes the card. One-agent path: `single.py` `SingleAgent`.
- `api/`: `production.py` `main()` serves the backend on 127.0.0.1:8000. The app itself is `app.py` `create_app()`.
- `cards/`: `train_lora.py` `train()` trains the card model (GPU only); `dataset.py` `build_examples()` builds its examples.
- `consistency/`: `train.py` `fit()` fine-tunes the consistency classifier; `model.py` `TrainedChecker` runs it.
- `context/`: `budget.py` `fit_context()` drops the oldest non-evidence sections until a token budget fits.
- `data/`: `loaders.py` `load_dataset()` reads a public dataset by name into conversations.
- `eval/`: `records.py` `write_run()` writes a run record; `metrics.py` holds the metric functions.
- `experiments/`: one script per experiment: `e1_retrieval.py` `main()`, `e2_context.py` `main()`, `e4_consistency.py` `main()`, `e9_study.py` `main(argv)`.
  `judge_check.py` `main()` is a runner, not an experiment: the cloud grader re-grades a sample of the local judge's claims.
- `guard/`: `citations.py` `guard_card()` keeps only findings whose citations are real turns that quote them.
- `llm/`: `client.py` `LLMClient` protocol; `OllamaClient` for local models, `FakeLLM` for tests.
- `memory/`: `store.py` `TranscriptStore` (in-memory and Postgres); `extract.py` `MemoryExtractor` builds memory items.
- `obs/`: `tracing.py` `Telemetry` keeps local spans and metrics for a press, with no transcript text.
- `prompts/`: `registry.py` `get_prompt(name, version)` loads `prompts/<version>/<name>.md`.
- `replay/`: `runner.py` `replay()` runs an assistant over scenarios and returns predictions.
- `retrieval/`: `hybrid.py` `Retriever` ranks turns by BM25 and embedding cosine similarity.
- `speech/`: `client.py` `GrpcSpeech` talks to the speech service; `diarize.py` `Diarizer` labels speakers.
- `vision/`: `local.py` `SceneNoter` makes an optional scene note with a local multimodal model; off by default.
- `tools/`: `taskctl.py` `main()` is the plan state machine (`status`, `next`, `done`, and more).
- `clients/web/`: `src/index.tsx` mounts `App` from `src/App.tsx`; `npm run dev` starts Vite, `npm test` runs Vitest.
- `clients/unity/`: `Assets/Scripts/ContextLensClient.cs` `ContextLensClient` (a MonoBehaviour); the socket is in `ContextLensSocket.cs`.

## Where to change common things

- Add a metric: write the function in `eval/metrics.py`, wire it into a section of `base_metrics()` in `eval/series.py`, and test it in `tests/test_metrics.py`.
- Add a prompt version: create `prompts/<version>/` with the `<name>.md` files, then pass the version to `get_prompt()`, `TeamAssistant` or `SingleAgent`. Prompt versions are saved in run records.
- Add a run record field: add the argument to `write_run()` in `eval/records.py`, then check it in `check()` in `eval/check_run.py`.
- Change the Ollama model: edit `MODEL` in `api/production.py`, `WRITER_MODEL` and `JUDGE_MODEL` in `experiments/e2_context.py`, and `PROMPTED_MODEL` in `consistency/select_cascade.py`; update `docs/model-choice.md`.
- Add a dataset: record its license in `docs/datasets.md` first (`tools/check_datasets_doc.py` checks that doc), then add a loader to `_LOADERS` in `data/loaders.py`, and to `_GOLD` if it has gold scenarios.
- Change a gate: task gate `cmd_done()` in `tools/taskctl.py`; run-record gate `check()` in `eval/check_run.py`; claims gate `check_claims()` in `eval/claims.py`.

## Fast tests

Run from the repo root, naming the test file you need:

`.venv/Scripts/python -m pytest -q tests/test_metrics.py`
