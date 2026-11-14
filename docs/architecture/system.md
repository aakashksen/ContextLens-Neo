# System overview

[Back to README](../../README.md) · Next: [Press lifecycle](press-lifecycle.md)

ContextLens Neo has five running parts: clients, one FastAPI backend, a C++ speech service, a store for turns and memory, and local models served by Ollama. The agents live inside the backend process. Experiments, training and the website sit outside the running product.

## Where everything runs

```mermaid
flowchart TB
    subgraph Clients ["Clients"]
        WEB["React and TypeScript<br/>clients/web"]
        UNI["Unity client<br/>clients/unity"]
    end

    subgraph Backend ["FastAPI backend, one Python process"]
        WS["WebSocket<br/>/ws/{conversation_id}"]
        CS["Conversation state<br/>diarizer, memory, summary"]
        ASM["Assistant<br/>team, single or<br/>team_shipping_changed"]
        GD["Guards<br/>citation, number, instruction"]
        TEL["Telemetry<br/>OpenTelemetry, Prometheus"]
    end

    SP["Speech service<br/>C++, whisper.cpp, gRPC"]
    DB[("PostgreSQL + pgvector<br/>turns and embeddings")]
    OL["Ollama<br/>qwen3.5:4b-q4_K_M<br/>nomic-embed-text"]
    PY["pyannote<br/>speaker model"]

    WEB <-->|"JSON frames"| WS
    UNI <-->|"JSON frames"| WS
    WS --> CS
    CS -->|"PCM chunks"| SP
    SP -->|"timed segments"| CS
    CS --> DB
    CS --> PY
    WS --> ASM
    ASM --> GD
    ASM <--> OL
    ASM --> DB
    ASM -.-> TEL
```

## Who owns what

| Part | Owns | Never does |
|---|---|---|
| Client | Recording indicator, pause, delete, the press button, rendering cards | Decide what a card says |
| WebSocket layer (`api/stream.py`) | One connection per client, the frozen view for each press, epochs that make delete safe | Let a press read a turn after its `as_of_seq` |
| Speech service (`speech/`) | Voice gating, utterances, whisper.cpp transcription | Know about speakers, conversations or cards |
| Transcript store | Turns, speakers, confidence, embeddings | Accept writes during a press (`_FrozenStore` raises) |
| Agents (`agents/`) | Searching, checking and drafting findings under a deadline | Cite anything the guards cannot verify |
| Guards (`guard/`, `agents/single.py`) | What reaches the card | Call a model for citation or number checks |
| Telemetry (`obs/tracing.py`) | Durations, token counts, bounded labels | Record transcript text |

## Data that crosses each boundary

```mermaid
flowchart LR
    C["Client"] -- "audio (base64 PCM), press, pause, resume, delete" --> B["Backend"]
    B -- "transcript, turn_update, card, status" --> C
    B -- "PCM chunks" --> S["Speech service"]
    S -- "text, start_ms, end_ms" --> B
    B -- "turn rows, vectors" --> D[("PostgreSQL")]
    B -- "frozen transcript text, tool results" --> L["Ollama"]
    L -- "JSON steps and findings" --> B
```

The message types are defined once, in `src/contextlens/api/contract.py`; the same file produces `docs/client-contract.schema.json`, which the Unity client is checked against (`tools/check_unity_schema.py`). Raw audio is never written to disk: the diarizer keeps only the most recent two minutes in memory. Live paths use local models only; a cloud model is used by experiments for grading and comparison, on benchmark text and invented scenarios.

## Three ways to run it

```mermaid
flowchart LR
    subgraph Fake ["tools/fake_server.py"]
        F1["Scripted turns and cards"]
    end
    subgraph Real ["tools/run_everything.py"]
        R1["Postgres, speech service,<br/>backend, web client"]
    end
    subgraph Site ["site/live.html"]
        S1["Scripted replay in the browser"]
    end
    CT["Client contract"]
    Fake --> CT
    Real --> CT
```

The fake server speaks the same contract with invented content, so a client can be built and tested with no models. `run_everything.py` starts the real stack and stops what it started on Ctrl-C. The website's live demo needs neither: it is a scripted replay inside the page, shaped after the fake server and the contract.

## Startup and readiness

The backend loads both models when it starts and keeps them loaded (a 30 minute `keep_alive`). Until both are loaded, `GET /ready` answers 503 and a WebSocket is closed with code 1013 (try again later), so a press never races a cold model load. If Ollama is down the loads are retried every 5 seconds and `/ready` names the failing model.

Related: [Press lifecycle](press-lifecycle.md), [Speech service](speech-service.md).
