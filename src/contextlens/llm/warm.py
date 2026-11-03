"""Loads the local models before the first press, so a cold load does not eat the press deadline."""

import threading
import time
from collections.abc import Callable


class Warmup:
    """Runs each loader once, in order, retrying a failed one. `ready` is set only when all have loaded;
    `status()` reports how long each took (the cold-start time) and what is failing."""

    def __init__(
        self, loaders: dict[str, Callable[[], None]], retry_s: float = 5.0,
        on_loaded: Callable[[str, float], None] | None = None,
    ) -> None:  # fmt: skip
        self.loaders, self.retry_s, self.on_loaded = loaders, retry_s, on_loaded
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ms: dict[str, int] = {}
        self._errors: dict[str, str] = {}

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="warmup").start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        for name, load in self.loaders.items():
            while not self._stop.is_set():
                t0 = time.perf_counter()
                try:
                    load()
                except Exception as exc:  # the model server may simply not be up yet
                    with self._lock:
                        self._errors[name] = type(exc).__name__
                    self._stop.wait(self.retry_s)
                    continue
                seconds = time.perf_counter() - t0
                with self._lock:
                    self._ms[name] = int(seconds * 1000)
                    self._errors.pop(name, None)
                if self.on_loaded:
                    self.on_loaded(name, seconds)
                break
            else:
                return
        self.ready.set()

    def status(self) -> dict:
        with self._lock:
            return {"ready": self.ready.is_set(), "cold_start_ms": dict(self._ms), "errors": dict(self._errors)}
