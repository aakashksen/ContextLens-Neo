# Client contract: `/ws/{conversation_id}`

Source of truth: `src/contextlens/api/contract.py`. Machine-readable schema:
`docs/client-contract.schema.json` (regenerate with
`python -m contextlens.api.contract > docs/client-contract.schema.json`).
Every message is one JSON text frame with a `type` field. Payload types
(`Turn`, `Card`, `ButtonRequest`) come from `src/contextlens/schemas.py`.

## Client to server

| type | fields | meaning |
|---|---|---|
| `audio` | `data` (string; intended as base64 of 16-bit mono PCM), `sample_rate` (integer, default 16000) | a chunk of speech |
| `press` | `request`: `ButtonRequest` | the button was pressed; a `card` is returned |
| `pause` | none | stop listening: `audio` is ignored (no reply) until `resume` |
| `resume` | none | start listening again |
| `delete` | none | delete the stored transcript; the server answers `deleted` and closes |

The fake server does not decode `audio.data`; any string is accepted and the
content is ignored. Each accepted `audio` message yields exactly one scripted
`transcript` turn (a real backend may differ). `press` is still answered while
paused. Only `request.conversation_id` and `as_of_seq` affect the fake's reply;
`question` and `detail` are accepted and ignored, and the card's `mode` is
always `recall`.

## Server to client

| type | fields | meaning |
|---|---|---|
| `transcript` | `turn`: `Turn` | a new transcribed turn |
| `card` | `card`: `Card` (status `answered`, `partial`, or `abstained`) | the answer to a `press` |
| `status` | `state`: `recording`, `paused`, `deleted`, `error`; optional `detail` | recording indicator and errors |

On connect the server sends `status` `recording`. A message that is not valid
is answered with `status` `error` (with a `detail`) and the connection stays
open. Invalid means any of: a binary frame (all messages must be text frames),
text that is not JSON, an unknown or missing `type`, or missing or wrongly typed
fields. A normal client disconnect just ends the session.

## State across connections

- The server keeps a conversation's turns (the ones it has sent as `transcript`)
  across reconnects to the same `conversation_id`. A new connection continues
  the sequence: the next `audio` yields the turn with the next `seq`, and a
  `press` can cite turns from an earlier connection. Seqs start at 0 for a
  conversation with no stored turns.
- `delete` clears those turns for that conversation, whichever connection
  sent them; a later connection starts again at `seq` 0.
- The press cycle (below) is per connection: it restarts at `answered` on every
  new connection, and is not reset by `pause`, `resume` or an error.
- `paused` is also per connection; a new connection starts recording.

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
returns the next scripted turn (the script has four turns and repeats).
The card status depends only on how many valid presses this connection has had,
not on the question: the 1st, 4th, 7th ... valid press returns `answered` (one
finding, plus a `suggestion`), the 2nd, 5th ... returns `partial` (one finding,
plus `missing`), and the 3rd, 6th ... returns `abstained` (no findings, plus
`missing`). Presses answered with `status` `error` do not count. To see all
three, a client sends at least one `audio` message, then presses three times
with a valid `as_of_seq` (for example the `seq` of the last transcript).
