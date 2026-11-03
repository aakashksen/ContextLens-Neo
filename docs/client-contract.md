# Client contract: `/ws/{conversation_id}`

Source of truth: `src/contextlens/api/contract.py`. Machine-readable schema:
`docs/client-contract.schema.json` (regenerate with
`python -m contextlens.api.contract > docs/client-contract.schema.json`).
Every message is one JSON text frame with a `type` field. Payload types
(`Turn`, `Card`, `ButtonRequest`) come from `src/contextlens/schemas.py`.

## Client to server

| type | fields | meaning |
|---|---|---|
| `audio` | `data` (base64 16-bit mono PCM), `sample_rate` (default 16000) | a chunk of speech |
| `press` | `request`: `ButtonRequest` | the button was pressed; a `card` is returned |
| `pause` | none | stop listening |
| `resume` | none | start listening again |
| `delete` | none | delete the stored transcript; the server answers `deleted` and closes |

## Server to client

| type | fields | meaning |
|---|---|---|
| `transcript` | `turn`: `Turn` | a new transcribed turn |
| `card` | `card`: `Card` (status `answered`, `partial`, or `abstained`) | the answer to a `press` |
| `status` | `state`: `recording`, `paused`, `deleted`, `error`; optional `detail` | recording indicator and errors |

On connect the server sends `status` `recording`. A malformed client message
gets `status` `error` and the connection stays open.

## Press rules

A `press` is answered from the turns the server has sent for this conversation
(`transcript` messages), never from turns the client has not received. Card
evidence cites only a sent turn with `seq <= as_of_seq` (the latest such turn),
with that turn's real speaker, text and start time.

- `request.conversation_id` differs from the URL's: `status` `error`, no card.
- `as_of_seq` negative: `status` `error`, no card.
- `as_of_seq` greater than the `seq` of the last turn sent, or no turn sent yet:
  `status` `error`, no card. Clients should press with the `seq` of the latest
  transcript turn they hold.
- An error leaves the connection open.
- After `delete`, nothing sent before it can be cited, even on a new connection.

## Fake server

`python tools/fake_server.py` serves this contract on `127.0.0.1:8000` (needs
`pip install -e .[server]`). Transcript text is invented. Each `audio` message
returns the next scripted turn; successive `press` messages return an
`answered`, then `partial`, then `abstained` card, repeating.
