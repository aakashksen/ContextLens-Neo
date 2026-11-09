# ContextLens

Press a button mid-conversation and get one short card based on what was actually said: something you
missed, a possible inconsistency, or help responding. Specialist agents do the digging, and every recalled
fact points back to the words it came from. "Not enough information" is a valid answer.

What was found, with every number tied to a run record, is in `docs/REPORT.md`. The short version: a single agent matched the
multi-agent team on every quality measure we can state, at a fraction of the tokens and latency, and prompting beat fine-tuning. The
design is `docs/DESIGN.md`, the reasoning behind each choice `docs/DECISIONS.md`, and how to defend them `docs/interview-notes.md`.

## Set up

Needs Windows or Linux, Python 3.12, Docker, Node.js, and [Ollama](https://ollama.com). A GPU is optional for the product and needed only to
reproduce training and the timed experiments.

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,server,postgres,speech,audio]"
ollama pull qwen3.5:4b-q4_K_M
ollama pull nomic-embed-text
npm --prefix clients/web install
```

Optional: put `HF_TOKEN=...` in a git-ignored `.env` for speaker labels (accept the terms of the pyannote model first, see `docs/datasets.md`).
Without it every turn keeps the speaker `unknown` and nothing else changes. No other keys are used. Never commit `.env`.

Check the install, offline and free (no GPU, no network, no model calls):

```
.venv\Scripts\python -m pytest -q
```

## Demo script

1. **Offline, no models.** The scripted server speaks the same WebSocket contract as the real backend with invented content:
   ```
   .venv\Scripts\python tools\fake_server.py
   npm --prefix clients/web run dev
   ```
   Open the address `npm` prints, with the server named in the address, as below. Four turns arrive; press the button and a card appears that cites the turn it relies on.
   ```
   http://127.0.0.1:5173/?server=ws://127.0.0.1:8000
   ```
2. **The real stack.** Start Ollama, then:
   ```
   py -3.12 tools\run_everything.py
   ```
   It starts Postgres with pgvector, the whisper.cpp speech service, the backend and the web client, and stops them on Ctrl-C
   (`docs/run-everything.md`). Open the same client address, allow the microphone, speak for a minute or two, and press "What was said?".
3. **What to look for.** Ask about something nobody said: the card should abstain. Say "make that Monday" after "Friday": the card should
   show both turns and call it an update, not a contradiction. A suggested reply is labelled as a suggestion.
4. **Switch the system behind the button** with `CONTEXTLENS_ASSISTANT` set to `team` (the default), `single` or `team_shipping_changed`
   before starting the backend. `docs/REPORT.md` says which of these was measured on the held-out test and which is the product default.

## Reproduce a result

Every experiment writes a record to `results/runs/`. `python -m contextlens.eval.claims check` re-derives each entry of `results/claims.json`
from its record, and `python tools/check_report_numbers.py` fails if the report or the bullets contain a number that is not in a cited record.
Experiments that need a GPU or a network are marked `real` and do not run by default.

## What's where

| Path | What it is |
|---|---|
| `src/contextlens/` | The agents, retrieval, guards, consistency checker, evaluation and experiments |
| `speech/` | The C++ whisper.cpp service (gRPC) |
| `clients/web/`, `clients/unity/` | The React/TypeScript client and the rebuilt Unity client |
| `docs/REPORT.md` | The report: problem, architecture, data, every experiment, limits |
| `docs/DESIGN.md`, `docs/DECISIONS.md` | What was built and why, with the options that lost |
| `docs/interview-notes.md` | Each decision, tradeoff and failure, for defending it |
| `docs/RESUME-TARGETS.md` | Resume bullets and the evidence each one rests on |
| `results/runs/`, `results/claims.json` | Experiment records, the only source of numbers, and the claims drawn from them |
| `legacy/OfficialContextLens/` | The hackathon code, unchanged and read-only |

## How the build ran

The build was driven by `docs/PLAN.md` through one orchestrator (`/run-plan` in Claude Code, `.claude/skills/run-plan/SKILL.md`):
each task is built test first in its own git worktree, finished only through `tools/taskctl.py`, which reruns the plan's acceptance commands.
This is how the project was built, not how to use it. To see where the plan stands:

```
py -3.12 tools\taskctl.py status
```

## GPU PyTorch (reranker and training)

The default Windows wheel is CPU-only. This machine's driver supports the recent CUDA wheels used below:

```
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install sentence-transformers
```
