# Run everything

One command starts the stack:

```
py -3.12 tools/run_everything.py
```

It starts, in this order, and stops what it started on Ctrl-C (volumes are kept):

| Part | How | Address |
|---|---|---|
| PostgreSQL + pgvector | `docker compose up -d db` | 127.0.0.1:5432 |
| Speech service (whisper.cpp, CPU) | `docker compose up -d speech` | localhost:50051 |
| Backend (`contextlens.api.production`) | Python, the real agent team | ws://127.0.0.1:8000/ws/{conversation_id} |
| Web client | `npm --prefix clients/web run dev` | http://127.0.0.1:5173/?server=ws://127.0.0.1:8000 |

A port that already answers is left alone, so a service you started yourself is used as it is. The first
`docker compose up` builds the speech image, which takes a while and needs the network.

## What you start yourself

- **Ollama**, serving `qwen3.5:4b-q4_K_M` and `nomic-embed-text` on `http://127.0.0.1:11434`
  (`ollama pull` both once). The script does not start Ollama or load a model. Without it the backend still
  starts, but turns cannot be stored (embedding) and presses return `status` `error`.
- **Speaker labels** need `HF_TOKEN` (environment or `.env`) from an account that accepted the terms of
  `pyannote/speaker-diarization-community-1` (see `docs/datasets.md`). Without it every turn stays `unknown`;
  nothing else changes. The speaker model is loaded once, on the CPU, and shared by all conversations; speech heard
  before it has loaded stays `unknown`.

Settings, all optional: `DATABASE_URL` (default is the compose database), `SPEECH_ADDR` (default
`localhost:50051`). Models are local only: the live path never calls Claude.

## What the backend does

- Audio goes to the speech service; each final segment becomes a turn, sent as `transcript` with speaker
  `unknown`. When the speaker model names the speaker, a `turn_update` follows (see `docs/client-contract.md`).
- A press runs the agent team on the turns up to `as_of_seq`, with a fresh context each time, and answers with
  a `card` whose evidence cites real turns.
- Delete removes the transcript, the memory items, any scene note and queued telemetry for the conversation.
- A conversation nobody has been connected to for a while is dropped from memory; its turns stay in the store and
  a later connection continues the sequence.
- Metrics sit in an in-process Prometheus registry. Serving them (`contextlens.obs.tracing.serve_metrics`) is not
  started here, because Prometheus in Docker needs a host address chosen on purpose (`deploy/prometheus.yml`).

## Tests

Offline, no GPU or network: `pytest tests/test_integration.py -q`. It runs the real backend with fake models.

With the services above running (`-m real`):

```
$env:PYTHONPATH = "src"
# real speech service, Postgres and pyannote; models faked (no Ollama):
.venv/Scripts/python -m pytest tests/test_integration.py -m real -k real_speech -q -s
# everything real, including Ollama (needs a GPU turn):
.venv/Scripts/python -m pytest tests/test_integration.py -m real -k test_press_end_to_end -q -s
```

The browser microphone path (`getUserMedia`, `AudioContext`, the live WebSocket) needs a person and a browser:
open the client URL above, allow the microphone, speak, and press "What was said?".
