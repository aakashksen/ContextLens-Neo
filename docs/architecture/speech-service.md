# Speech service

[Back to README](../../README.md) · Previous: [Retrieval and memory](retrieval-and-memory.md) · Next: [Evaluation and claims](evaluation-and-claims.md)

Speech goes in as 16 kHz mono 16-bit PCM and comes out as timed text segments, which the backend turns into transcript turns. A separate model names the speakers later, in the background. Source: `speech/` (C++), `src/contextlens/speech/` and `api/stream.py` (Python).

## The path of one utterance

```mermaid
flowchart LR
    MIC["Client microphone"] -->|"audio message<br/>base64 PCM"| WS["Backend WebSocket"]
    WS --> V["Validate<br/>16 kHz, base64,<br/>whole samples"]
    V --> DZ["Diarizer buffer<br/>last two minutes,<br/>in memory only"]
    V --> GR["gRPC stream<br/>Speech.Transcribe"]
    GR --> SG["Segmenter<br/>voice gate"]
    SG --> WH["whisper.cpp<br/>ggml tiny.en"]
    WH --> SEG["TranscriptSegment<br/>text, start_ms, end_ms"]
    SEG --> TURN["Turn, speaker unknown"]
    TURN --> ST[("Transcript store")]
    TURN --> CL["Client: transcript message"]
```

## The C++ service

```mermaid
flowchart TB
    subgraph SVC ["speech/src: one gRPC server"]
        MAIN["main.cpp<br/>Transcribe(stream AudioChunk)<br/>returns stream TranscriptSegment"]
        VAD["vad.cpp: VoiceGate and Segmenter"]
        RB["ring_buffer.cpp<br/>pre-roll samples"]
        TR["transcriber.cpp<br/>one shared whisper context,<br/>calls serialised"]
    end
    MAIN -->|"int16 to float"| VAD
    VAD --> RB
    VAD -->|"Utterance"| TR
    TR -->|"Segment list"| MAIN
```

- **Contract** (`speech/proto/speech.proto`): a bidirectional stream. The client sends `AudioChunk{pcm}` of any size; the server sends `TranscriptSegment{text, start_ms, end_ms, is_final}` with times in milliseconds from the start of the stream, and reports its compute backend (`cpu` or `gpu:<name>`) as initial metadata under `x-compute-backend`. Every segment the service emits is final.
- **Voice gate** (`vad.cpp`): an energy gate on 20 ms frames. A frame is loud above a fixed RMS threshold, the gate stays open through a hangover after the last loud frame, a short pre-roll from a ring buffer is kept so word starts are not clipped, an utterance with too little loud audio is dropped, and a long one is cut at a maximum length. The settings are in `SegmenterConfig`.
- **Transcriber** (`transcriber.cpp`): one whisper.cpp model shared by every stream, calls serialised by a mutex. Segment times are shifted by the utterance start so they are relative to the whole stream.
- **Build** (`speech/Dockerfile`, `CMakeLists.txt`): C++17, gRPC and protobuf, whisper.cpp pinned to a release tag and built from source, a model file downloaded at a pinned revision and checked against a SHA-256, and `ctest` run during the image build so a broken build cannot produce an image. A GPU variant is the same file with CUDA base images (`docker compose --profile gpu`), and the service reports `gpu:<name>` only if whisper.cpp actually initialised that GPU.
- **Tests**: `speech/tests/test_core.cpp` covers the gate, the segmenter and the ring buffer without a model.

## The Python side

```mermaid
sequenceDiagram
    autonumber
    participant C as Connection
    participant G as GrpcSpeech
    participant S as Speech service
    participant D as Diarizer
    C->>G: open() with a timeout
    G->>S: connect, wait for channel ready
    loop every audio message
        C->>D: feed(pcm)
        C->>G: send(pcm)
        G->>S: AudioChunk
    end
    S-->>G: TranscriptSegment
    G-->>C: segment
    C->>C: ingest under the lock, only if the epoch and stream are current
    C-->>C: transcript message to every client
```

`GrpcSpeech` (`speech/client.py`) encodes and decodes the two protobuf messages by hand over a generic gRPC channel and bounds its send queue. If the service does not answer in time, or fails, the connection closes the stream, tells the client once that "audio is not being transcribed", and retries after a pause (10 s by default); the rest of the backend keeps working. `pause` makes the server ignore audio until `resume`, and a delete closes the stream so audio from before it cannot come back. A press is still answered while paused.

## Speaker labels

```mermaid
flowchart TB
    T["New turn stored<br/>speaker unknown, confidence 0"] --> Q["schedule_speakers"]
    Q --> W["Diarizer window<br/>last two minutes of audio"]
    W --> P["pyannote pipeline<br/>spans: start, end, speaker"]
    P --> M["Match each span's name to the<br/>stable name already given to<br/>turns it overlaps"]
    M --> U["Per turn: the speaker with the<br/>largest overlap, confidence =<br/>its share of the turn"]
    U --> R["relabel in the store"]
    R --> N["turn_update to clients<br/>same id and seq: replace, do not append"]
```

The pipeline names speakers afresh on every run, so `Diarizer.speaker_updates` maps its names onto stable ones (`speaker_1`, `speaker_2`, and so on) by overlap before relabelling. The speaker model is one shared instance, loaded once on a background thread, and runs serialised; speech heard before it has loaded stays `unknown`. It needs a Hugging Face token from an account that accepted the model's terms (`HF_TOKEN` in a git-ignored `.env`); without it every turn stays `unknown` and nothing else changes. Retrieval treats a speaker with confidence below 0.5 as `unknown`, so a weak label is never presented as fact.

## What the speech experiments found

E7 streamed whole AMI meetings through this service at four times real speed. The streamed word error rate was at first far worse than offline Whisper on the same audio. An offline control showed why: the fixed loudness gate dropped most quiet speech. Rebuilt meetings were given one gain each before streaming, which brought streamed word error rate to 0.376 and speaker error to 0.599 on AMI's published labels. That gain is a normalization of rebuilt test audio, computed over a whole meeting, and a live system cannot do it. The fixed gate in the product still drops quiet microphones, and an adaptive gate is listed in `docs/backlog.md`. See [Evaluation and claims](evaluation-and-claims.md) and [REPORT](../REPORT.md), E7.

Related: [System overview](system.md), [Retrieval and memory](retrieval-and-memory.md).
