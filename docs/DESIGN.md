# ContextLens design

## Purpose
ContextLens is an on-demand conversation assistant. During a conversation, the user presses a button and gets one short card built
from what was actually said. It's for anyone who loses track of details in meetings, classes, or long discussions. Neurodivergent users
are one audience the accessibility settings serve; they aren't the only audience.

Central question: **How can an assistant keep and find the right context quickly enough to help, without inventing memories or
mistaking changed information for a contradiction?**

The project also exists to demonstrate specific skills for the resume (`docs/RESUME-TARGETS.md`), so every feature must serve the product
*and* produce a measured result. Anything that serves neither is out.

## What the user sees

| Mode | Situation | Card |
|---|---|---|
| **Recall** | "What was the reason we ruled that out?" | "Maya said it exceeded the budget." + source at 04:32 |
| **Clarify** | A deadline differs from an earlier one | "Earlier: Monday. Now: Wednesday. No one said it changed." + both sources |
| **Respond** | Someone asks you something after a long discussion | "You prioritized cost earlier." + a suggested reply, labeled as a suggestion |

- **Default press** means "help me with what's happening now". A typed question is optional.
- **Remembered vs suggested** must look different. A suggestion must never look like something the user said.
- **Not enough information** is a valid answer. A wrong answer is worse than none.
- **Conflicts are flagged with evidence**, never as accusations. Tone or sarcasm readings are worded as tentative.
- **Settings:** detail level (brief, standard, detailed), plain wording, and text size.
- **Consent:** a visible recording indicator, a pause control, and a delete-transcript action. Raw audio isn't stored by default.

## Architecture

```
 Audio ─> C++ whisper.cpp service ──gRPC──> Streaming backend (FastAPI) ─┬─> Transcript store (PostgreSQL)
          + speaker diarization                                         ├─> Index (BM25 + pgvector)
                                                                        └─> Memory extractor (decisions, commitments,
 Camera (optional) ─> local vision ─> scene note                                questions, corrections, with supersedes links)

 Button ─> Controller: freeze the conversation at as_of_seq
              ├─ Memory agent        (search, surrounding turns, memory lookup)
              ├─ Consistency agent   (search, consistency check)
              └─ Assistance agent    (writes the card; may send one specialist back once)
                     │  shared evidence store (turn references)
                     ▼
              Citation guard: drops claims without a real source, abstains if nothing is left
                     ▼
              Card over WebSocket ─> React/TypeScript client, Unity XR client

 Everywhere: OpenTelemetry traces, Prometheus metrics, Grafana dashboard, token and cost counts
```

## Decisions

| Area | Decision | Reason |
|---|---|---|
| Hackathon services | Deepgram, Gemini and Vercel removed; old code kept in `legacy/` | Use only what's available to us |
| Speech | whisper.cpp, in a C++ service built in Docker, over gRPC | Local and private, and real C++ work |
| Speakers | Local diarization (pyannote, or another local option if its terms don't fit) | Removes the Deepgram dependency |
| Models | Ollama by default (`gemma4`, `nomic-embed-text` are installed); vLLM behind the same interface; DeepSeek as the cloud comparison | Available now and free locally, with an option to compare |
| Vision | Optional local vision, off by default | Multimodal input, but not core to the card |
| Backend | Python 3.12 and FastAPI | Same language as the agents and training |
| Agents | LangGraph | Asked for by roles, and suits parallel work and follow-ups |
| Storage | PostgreSQL + pgvector (Docker), BM25 through PostgreSQL full-text search or `rank-bm25` | One store; SQL evidence |
| Reranking | Local cross-encoder | Only claimed if it measurably helps |
| Clients | React/TypeScript ported from legacy; Unity 6 client rebuilt | The original frontend exists; Unity code was lost |
| Training | PyTorch, Hugging Face Transformers, PEFT (QLoRA) | Standard, and fits an 8 GB GPU in 4-bit |
| Experiments | MLflow for tracking, run records in `results/runs/` for claims | Reproducibility |

## Limits
- **Hardware:** RTX 4070 laptop GPU with 8 GB VRAM. Only one GPU job at a time. If a run can't fit, block the task; don't quietly shrink what the bullet says.
- **Cloud spending:** $5 total across all cloud model calls, tracked in `results/spend.json`. `LLMClient` refuses past the cap. Only the person can raise it. Renting a GPU requires their approval.
- **Data leaving the machine:** only benchmark text and synthetic scenarios may go to a cloud model. Live audio, transcripts, and study data stay local.

## Agents

| Agent | Job | Tools | Failure to measure |
|---|---|---|---|
| **Controller** | Freezes the state, runs the specialists, enforces the deadline | None | Deadline misses |
| **Memory** | Finds relevant earlier information and who said it | `search_transcript`, `get_turns`, `lookup_memory` | Missed evidence, wrong speaker |
| **Consistency** | Separates contradiction from a legitimate update, and from too little evidence | `search_transcript`, `check_consistency` | False alarms |
| **Assistance** | Writes the card, and may ask one specialist one follow-up | Shared evidence only | Invented details, misleading suggestions |

The team runs under a deadline. When it expires, the controller returns whatever verified findings exist as a `partial` card. The design
must be compared to a single agent with the same tools on quality, latency, and cost. If the single agent wins, that gets reported, and the
architecture bullet changes.

## Non-goals
Always-on interruptions, identity or emotion detection of people, fact-checking against the internet, long-term memory across
different conversations, mobile apps, public hosting, and accounts. Rust, Kafka, Redis, Spark, and Terraform are left out because the product doesn't need them.

## Anti-goals: this project has failed if
- A number on the resume can't be traced to a run record.
- A card puts words in someone's mouth, or presents a suggestion as something that was said.
- It flags a legitimate correction as a contradiction often enough that users stop trusting it, however good the architecture sounds.
- The person can't explain and defend how any part works in an interview.

## Left to the builder
Library versions, internal helpers, chunk sizes to try (listed by experiment), and model candidates within the stated limits, as long as
choices are recorded in the run records.
