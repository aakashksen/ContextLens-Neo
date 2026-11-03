"""set_backend swaps the backend and its background flag together."""

from contextlens.speech.client import Segment
from contextlens.speech.diarize import Diarizer


def test_assign_never_sees_a_backend_without_its_flag():
    d = Diarizer("c1")
    d.new_stream()
    seen = {}

    class WindowLike:
        @property
        def spans(self):  # read by set_backend while it is switching; assign runs "on another thread" here
            try:
                d.assign([Segment(text="x", start_ms=0, end_ms=500)])
            except Exception as e:
                seen["err"] = type(e).__name__
            return lambda pcm: []

    d.set_backend(WindowLike())
    assert seen == {}
    assert d.background is True
