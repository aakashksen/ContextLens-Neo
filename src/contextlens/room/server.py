"""The meeting room's HTTP side. `add_room(app, ...)` puts these routes on a FastAPI app:

  GET  /api/health                      mode (replay or live), assistants, credit line
  GET  /api/room/meetings               the meetings with audio
  GET  /api/room/meetings/{id}/replay   timed transcript and annotation-based memory (replay mode only)
  GET  /api/room/audio/{id}, /video/{id}  the media, with byte ranges so the player can seek
  POST /api/room/press                  cards and traces; the server enforces what was said before the press
  POST /api/room/live/start|control|stop, GET /api/room/live/{sid}/events    live mode only

The same app can carry the real /ws/{conversation_id} (live mode adds it), and serves site/ last."""

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.replay.runner import Context
from contextlens.room.live import LiveRig, LiveSession, poll_forever
from contextlens.room.meetings import ATTRIBUTION, Meeting, align_items, find_meetings, parse_range, summary_items, timed_turns, valid_id
from contextlens.room.replay import ReplayAssistant, as_of_seq, memory_json, spoken_by, turn_json
from contextlens.room.trace import build_trace
from contextlens.schemas import ButtonRequest

ASSISTANTS = ("single_plus", "team_routed")  # the two arms the room compares; names from api/production.py
CHOICES = (*ASSISTANTS, "both")
MAX_QUESTION = 300
MAX_SESSIONS = 4
CHUNK = 1 << 20


class PressBody(BaseModel):
    meeting: str
    assistant: str = "both"
    question: str | None = Field(default=None, max_length=MAX_QUESTION)
    position_ms: int = Field(default=0, ge=0)  # replay: where the player is; live: ignored, the stored transcript is used
    session: str | None = None


class StartBody(BaseModel):
    meeting: str
    position_ms: int = Field(default=0, ge=0)


class ControlBody(BaseModel):
    action: str
    position_ms: int = Field(default=0, ge=0)
    playing: bool = False


def names_for(choice: str) -> list[str]:
    if choice not in CHOICES:
        raise HTTPException(400, f"assistant must be one of {CHOICES}")
    return list(ASSISTANTS) if choice == "both" else [choice]


class ReplayData:
    """One meeting's replay material, read once: the timed transcript and the annotation-based memory."""

    def __init__(self, raw: Path, meeting_id: str) -> None:
        self.turns = timed_turns(raw, meeting_id)
        self.items = align_items(meeting_id, summary_items(raw, meeting_id), self.turns)
        self.by_seq = {t.seq: t for t in self.turns}

    def payload(self) -> dict:
        return {"turns": [turn_json(t) for t in self.turns],
                "memory": [memory_json(m, self.by_seq[m.as_of_seq].end_ms) for m in self.items],
                "flags": [],
                "memory_source": "AMI annotators' decisions and actions, placed by word overlap (no memory model in replay)",
                "captions_source": "AMI transcript with word timings (no speech model in replay)"}


def serve_file(path: Path, request: Request, media_type: str) -> Response:
    """The file, or the one byte range asked for, streamed in pieces so a long meeting is never held in memory."""
    size = path.stat().st_size
    try:
        rng = parse_range(request.headers.get("range"), size)
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    start, end = rng if rng else (0, size - 1)

    def pieces():
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0 and (block := f.read(min(CHUNK, left))):
                left -= len(block)
                yield block

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1), "Cache-Control": "no-store"}
    if rng:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(pieces(), status_code=206 if rng else 200, media_type=media_type, headers=headers)


def add_room(app: FastAPI, raw: Path, *, rig: LiveRig | None = None, replay_assistant: Callable = ReplayAssistant,
             site: Path | None = None, poll_s: float = 1.0, pace=asyncio.sleep) -> None:
    meetings: dict[str, Meeting] = find_meetings(raw)
    replay_cache: dict[str, ReplayData] = {}
    sessions: dict[str, LiveSession] = {}
    pollers: dict[str, asyncio.Task] = {}
    live = rig is not None

    def meeting(mid: str) -> Meeting:
        if not valid_id(mid) or mid not in meetings:
            raise HTTPException(404, "no such meeting")
        return meetings[mid]

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "mode": "live" if live else "replay", "assistants": list(ASSISTANTS), "attribution": ATTRIBUTION,
                "models": live, "captions": "speech service" if live else "AMI transcript"}

    @app.get("/api/room/meetings")
    def list_meetings() -> dict:
        return {"meetings": [m.public() for m in meetings.values()], "attribution": ATTRIBUTION}

    @app.get("/api/room/meetings/{mid}/replay")
    def replay(mid: str) -> dict:
        meeting(mid)
        if live:
            raise HTTPException(409, "live mode: captions come from the speech service, not a stored transcript")
        if mid not in replay_cache:
            replay_cache[mid] = ReplayData(raw, mid)
        return {"id": mid, **replay_cache[mid].payload()}

    @app.get("/api/room/audio/{mid}")
    def audio(mid: str, request: Request) -> Response:
        return serve_file(meeting(mid).wav, request, "audio/wav")

    @app.get("/api/room/video/{mid}")
    def video(mid: str, request: Request) -> Response:
        v = meeting(mid).video
        if v is None:
            raise HTTPException(404, "no video for this meeting")
        return serve_file(v, request, "video/webm" if v.suffix == ".webm" else "video/mp4")

    def replay_press(body: PressBody, names: list[str]) -> dict:
        if body.meeting not in replay_cache:
            meeting(body.meeting)
            replay_cache[body.meeting] = ReplayData(raw, body.meeting)
        data = replay_cache[body.meeting]
        seq = as_of_seq(data.turns, body.position_ms)
        if seq < 0:
            return {"as_of_seq": -1, "position_ms": body.position_ms, "results": [], "error": "nothing has been said yet"}
        store = InMemoryTranscriptStore()
        for t in spoken_by(data.turns, body.position_ms):  # the store never holds a turn the press may not use
            store.add(t)
        results = []
        for name in names:
            ctx = Context(store=store, as_of_seq=seq)
            req = ButtonRequest(conversation_id=body.meeting, as_of_seq=seq, question=(body.question or "").strip() or None)
            card = replay_assistant(name).answer(req, ctx)
            results.append({"assistant": name, "card": card.model_dump(mode="json"),
                            "trace": build_trace(name, card, ctx.stage_ms, ctx.drops, speech_note="AMI transcript stands in for speech")})
        return {"as_of_seq": seq, "position_ms": body.position_ms, "results": results, "simulated": True}

    @app.post("/api/room/press")
    def press(body: PressBody) -> dict:
        names = names_for(body.assistant)
        question = (body.question or "").strip() or None
        if not live:
            return replay_press(body, names)
        s = sessions.get(body.session or "")
        if s is None:
            raise HTTPException(404, "no such session")
        return s.press(names, question)

    def session(sid: str) -> LiveSession:
        if not live:
            raise HTTPException(409, "the live pipeline is off; start the server with --live")
        if sid not in sessions:
            raise HTTPException(404, "no such session")
        return sessions[sid]

    @app.post("/api/room/live/start")
    async def live_start(body: StartBody) -> dict:
        if not live:
            raise HTTPException(409, "the live pipeline is off; start the server with --live")
        m = meeting(body.meeting)
        if len(sessions) >= MAX_SESSIONS:
            await live_close(next(iter(sessions)))
        sid = uuid.uuid4().hex[:8]
        s = sessions[sid] = LiveSession(sid, m.id, m.wav, m.duration_ms, rig, pace=pace)
        pollers[sid] = asyncio.get_running_loop().create_task(poll_forever(s, poll_s))
        return {"session": sid}

    @app.post("/api/room/live/{sid}/control")
    async def live_control(sid: str, body: ControlBody) -> dict:
        s = session(sid)
        try:
            if body.action == "play":
                await s.play(body.position_ms)
            elif body.action == "pause":
                await s.pause()
            elif body.action == "seek":
                await s.seek(body.position_ms, body.playing)
            else:
                raise HTTPException(400, "action must be play, pause or seek")
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse({"error": f"the pipeline failed ({type(exc).__name__})"}, status_code=502)
        return {"ok": True}

    @app.get("/api/room/live/{sid}/events")
    def live_events(sid: str, since: int = 0) -> dict:
        events, nxt = session(sid).log.since(max(since, 0))
        return {"events": events, "next": nxt}

    @app.delete("/api/room/live/{sid}")
    async def live_close(sid: str) -> dict:
        s = session(sid)
        task = pollers.pop(sid, None)
        if task:
            task.cancel()
        sessions.pop(sid, None)
        await s.close()
        return {"ok": True}

    if site is not None:
        app.mount("/", StaticFiles(directory=str(site), html=True), name="site")
