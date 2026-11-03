"""An optional scene note from a local multimodal model. Off by default; nothing runs unless enabled.

The note is context only: it is untrusted text, never evidence, and any failure just means no note.
Image bytes go to the model call and nowhere else (not disk, not logs, not the note)."""

import base64
import itertools
import threading

from pydantic import BaseModel

from contextlens.guard.citations import SCENE_PREFIX
from contextlens.llm.client import LLMClient, OllamaClient
from contextlens.llm.untrusted import escape

MODEL = "qwen3.5:4b-q4_K_M"
MAX_NOTE_CHARS = 300
_ids = itertools.count(1)

_PROMPT = "In one short sentence, describe what is visible in this image. Describe only what you see."


class SceneNote(BaseModel):
    id: str
    conversation_id: str
    text: str


def scene_block(note: SceneNote) -> str:
    """The one place a scene note is put into a prompt: escaped, delimited, one line."""
    return f"<scene_note>{escape(note.text)}</scene_note>"


class SceneStore:
    """The latest scene note per conversation, in memory only."""

    def __init__(self) -> None:
        self._latest: dict[str, SceneNote] = {}

    def put(self, conversation_id: str, note: SceneNote) -> None:
        self._latest[conversation_id] = note

    def latest(self, conversation_id: str) -> SceneNote | None:
        return self._latest.get(conversation_id)

    def delete(self, conversation_id: str) -> None:
        """Forget the note. T8.2's delete calls this next to the transcript store's delete; T8.7 wires it."""
        self._latest.pop(conversation_id, None)


class SceneNoter:
    def __init__(
        self, llm: LLMClient | None = None, enabled: bool = False, timeout_s: float = 3.0
    ) -> None:
        self.enabled, self.timeout_s = enabled, timeout_s
        self._llm = llm

    def _client(self) -> LLMClient:
        if self._llm is None:
            self._llm = OllamaClient(MODEL)
        return self._llm

    def note(self, image: bytes, conversation_id: str) -> SceneNote | None:
        """A short note, or None when disabled, failed, or too slow. Waits at most timeout_s."""
        if not self.enabled:
            return None
        box: list[str] = []

        def run() -> None:
            try:
                msg = {"role": "user", "content": _PROMPT, "images": [base64.b64encode(image).decode()]}
                res = self._client().complete([msg], max_tokens=80, timeout_s=self.timeout_s)
                box.append(res.text)
            except Exception:
                pass

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(self.timeout_s)
        text = " ".join(box[0].split())[:MAX_NOTE_CHARS] if box else ""
        if not text:
            return None
        return SceneNote(id=f"{SCENE_PREFIX}{next(_ids)}", conversation_id=conversation_id, text=text)
