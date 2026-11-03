# Run everything

One command starts the stack:

```
py -3.12 tools/run_everything.py
```

It starts, in this order, and stops what it started on Ctrl-C, on an error, or when a part exits (whole process
trees, so nothing keeps holding port 5173 or 8000; volumes are kept). It runs the backend with the project's
`.venv`, and says so at once if the backend exits right after starting:

| Part | How | Address |
|---|---|---|
| PostgreSQL + pgvector | `docker compose up -d db` | 127.0.0.1:5432 |
| Speech service (whisper.cpp, CPU) | `docker compose up -d speech` | 127.0.0.1:50051 |
| Backend (`contextlens.api.production`) | Python, the real agent team | ws://127.0.0.1:8000/ws/{conversation_id} |
| Web client | `npm --prefix clients/web run dev` | http://127.0.0.1:5173/?server=ws://127.0.0.1:8000 |

Every published Docker port is bound to 127.0.0.1 (the database holds transcripts, the speech service receives raw
audio), including the opt-in GPU speech service on 50052. A port that already answers is left alone, so a service you started yourself is used as it is. The first
`docker compose up` builds the speech image, which takes a while and needs the network.

## What you start yourself

- **Ollama**, serving `qwen3.5:4b-q4_K_M` and `nomic-embed-text` on `http://127.0.0.1:11434`
  (`ollama pull` both once). The script does not start Ollama or load a model. Without it the backend starts
  but stays not ready (see Warm-up) and accepts no connection.
- **Speaker labels** need `HF_TOKEN` (environment or `.env`) from an account that accepted the terms of
  `pyannote/speaker-diarization-community-1` (see `docs/datasets.md`). Without it every turn stays `unknown`;
  nothing else changes. The speaker model is loaded once, on the CPU, and shared by all conversations; speech heard
  before it has loaded stays `unknown`.

Settings, all optional: `DATABASE_URL` (default is the compose database), `SPEECH_ADDR` (default
`localhost:50051`). Models are local only: the live path never calls Claude.

## Warm-up

The backend loads both models (`qwen3.5:4b-q4_K_M` and `nomic-embed-text`) when it starts, with Ollama's
`keep_alive` of 30 minutes on every request so they stay loaded between presses. Until both are loaded,
`GET http://127.0.0.1:8000/ready` answers 503 and a WebSocket connection is closed with code 1013 (try again
later); `tools/run_everything.py` waits for it and prints the load times. When it answers 200 its body holds
`cold_start_ms` for each model, and the same times are the Prometheus gauge `contextlens_warmup_seconds{model}`.
If Ollama is down the loads are retried every 5 seconds and `/ready` lists the failing model under `errors`.

## What the backend does

- Audio goes to the speech service; each final segment becomes a turn, sent as `transcript` with speaker
  `unknown`. When the speaker model names the speaker, a `turn_update` follows (see `docs/client-contract.md`).
- A press runs the agent team on the turns up to `as_of_seq`, with a fresh context each time, and answers with
  a `card` whose evidence cites real turns.
- Delete removes the transcript, the memory items, any scene note and queued telemetry for the conversation.
- A conversation nobody has been connected to for a while is dropped from memory; its turns stay in the store and
  a later connection continues the sequence.
- Metrics sit in an in-process Prometheus registry and are off by default; see Metrics (opt-in).

## Metrics (opt-in)

Nothing here starts Prometheus or Grafana, and the backend serves no metrics unless asked. To turn them on:

```
$env:GRAFANA_ADMIN_PASSWORD = "<a password you choose>"     # Grafana refuses to start without one
docker compose --profile obs up -d prometheus grafana
$env:CONTEXTLENS_METRICS_PORT = "9464"                       # then start the backend; it serves /metrics there
```

Prometheus (127.0.0.1:9090) and Grafana (127.0.0.1:3000) are published on the loopback only. Prometheus reaches
the backend as `host.docker.internal:9464` (`deploy/prometheus.yml`); the backend serves `/metrics` on
`CONTEXTLENS_METRICS_HOST` (default 127.0.0.1, which Docker Desktop can reach; on Linux set it to the Docker
bridge address, never `0.0.0.0`). Stop with `docker compose --profile obs stop prometheus grafana`.

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

Browser path (`-m real`, needs Edge, Node, `clients/web/node_modules` and the CPU speech service on 50051):

```
.venv/Scripts/python -m pytest tests/test_browser.py -m real -q -s
```

Headless Edge plays `tests/fixtures/speech/phrase.wav` as its microphone into the real web client
(`getUserMedia`, `AudioContext`, the WebSocket), the real backend and the real speech service, then presses and
reads the card off the page. The language models are fake there. It does not check a human voice in a room, a
physical microphone, browsers other than Edge, or the real models; for those, open the client URL above, allow
the microphone, speak, and press "What was said?".

The Prometheus path (`-m real`, needs Docker; stops what it starts and removes no volume):

```
$env:GRAFANA_ADMIN_PASSWORD = "<throwaway>"
.venv/Scripts/python -m pytest tests/test_metrics_stack.py -m real -q -s
```

