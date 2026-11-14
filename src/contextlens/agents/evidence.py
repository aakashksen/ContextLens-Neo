import threading

from contextlens.schemas import Finding


class EvidenceStore:
    """The only thing agents share: verified findings. Safe to add to from parallel agents."""

    def __init__(self) -> None:
        self._findings: list[Finding] = []
        self._lock = threading.Lock()

    def add(self, finding: Finding) -> None:
        with self._lock:
            if finding not in self._findings:  # a follow-up may find again what the first pass found
                self._findings.append(finding)

    def all(self) -> list[Finding]:
        with self._lock:
            return list(self._findings)
