# Press lifecycle

[Back to README](../../README.md) · Previous: [System overview](system.md) · Next: [Agents and guards](agents-and-guards.md)

A press is one request: "what was actually said?" These diagrams follow it from the button to the card, through `api/stream.py`, `api/production.py`, `agents/graph.py` and `agents/single.py`.

## 1. Press to card

```mermaid
sequenceDiagram
    autonumber
    actor U as Person
    participant C as Client
    participant B as Connection
    participant S as Frozen store
    participant A as Assistant
    participant G as Guards

    U->>C: presses the button
    C->>B: press {conversation_id, as_of_seq, question?, detail}
    B->>B: validate: same conversation, as_of_seq not negative,<br/>not past the last turn sent
    B->>S: freeze the store at as_of_seq (read only)
    B->>A: answer(request, context) in a worker thread
    A->>S: tools read turns, never past as_of_seq
    A->>G: draft card with findings and quotes
    G-->>A: kept findings, or abstained
    A-->>B: card with latency, tokens and cost
    B->>B: every cited turn must be inside the frozen range
    B-->>C: card message
    C-->>U: one short card
```

Invalid presses are answered with a `status` of `error` and a reason. A card that cites a turn outside the frozen range is refused before it is sent. The whole answer has a hard limit of 60 seconds (`answer_timeout_s`) on top of the assistant's own 8 s deadline.

## 2. The frozen view

```mermaid
flowchart LR
    T0["turns 0 .. N<br/>spoken before the press"] --> FZ["_FrozenStore<br/>reads capped at as_of_seq"]
    T1["turns N+1 ...<br/>spoken after the press"] -.->|"unreachable"| FZ
    FZ --> TL["Tools: search, get turns,<br/>lookup memory"]
    FZ --> GU["Guards: resolve citations"]
```

Every retrieval and tool call filters on `as_of_seq`, and the store wrapper raises on any write, so a press cannot change the transcript or see a later turn. A test confirms that a later turn cannot appear in evidence.

## 3. Inside the assistant, in time

```mermaid
sequenceDiagram
    autonumber
    participant K as Controller
    participant M as Memory agent
    participant X as Consistency agent
    participant E as Evidence store
    participant Q as Assistance agent
    participant G as Guards

    K->>K: freeze the latest turns
    par parallel specialists
        K->>M: search, get turns, lookup memory (up to 3 steps)
        M->>E: verified findings
    and
        K->>X: search, check consistency (up to 3 steps)
        X->>E: verified findings
    end
    K->>Q: findings only, never the transcript
    Q->>Q: choose mode: recall, clarify or respond
    opt one follow-up allowed
        Q->>M: look again for something specific
        M->>E: more findings
    end
    Q->>G: card
    G-->>K: checked card
    K->>K: status: answered, partial or abstained
```

Specialists stop at the deadline; the controller returns whatever verified findings exist as a `partial` card. The single-agent system replaces the specialists and the assistance step with one tool loop of up to four steps and the same guards.

## 4. The deadline

```mermaid
flowchart LR
    S["press"] --> W["agents work<br/>deadline 8 s"]
    W -->|"all done"| C["card"]
    W -->|"deadline reached"| P["partial card<br/>with what is verified"]
    W -->|"a call blocks"| M["margin 0.5 s:<br/>the abandoned call is<br/>never an answer"]
    M --> P
```

A model call that blocks past the deadline is abandoned (at most 8 are left running at once, after which no new call starts), and anything it returns later is not used. The card records tokens and cost, and notes any abandoned call so the totals are known to be incomplete.

## 5. Card states

```mermaid
stateDiagram-v2
    [*] --> Frozen: press accepted
    Frozen --> Abstained: no turns, store failed or timed out
    Frozen --> Working: turns found
    Working --> Abstained: no finding passes the guards
    Working --> Partial: deadline, a failed agent, or some findings dropped
    Working --> Answered: every finding verified
    Answered --> [*]
    Partial --> [*]
    Abstained --> [*]
```

A card carries a `mode` (`recall`, `clarify` or `respond`), a `status`, its findings with quoted evidence, an optional suggestion, `missing` notes saying what was dropped and why (counts only), and its latency, tokens and cost. In respond mode a second model call writes the suggestion from the surviving statements only, and a number in it must also be in the quotes. A press with no question, on a latest turn that asks something (a question mark, or "please"), is set to respond in code, not by the model's guess.

Related: [Agents and guards](agents-and-guards.md), [Retrieval and memory](retrieval-and-memory.md).
