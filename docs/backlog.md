# Backlog

Product limits found while measuring, not yet planned. Nothing here is a plan task.

## An adaptive or causal voice gate for the speech service

Found in E7 (2026-11-06, `docs/DECISIONS.md` entry 68). The speech service (`speech/src/vad.h`) opens its gate on a fixed
RMS threshold of 0.01 per 20 ms frame. A quiet microphone, or any meeting recorded below that level, loses most of its speech
before Whisper sees it: on one AMI meeting the gate was open over 7.9% of the reference words and the streamed text held 7.4%
of them, against a WER of 0.26 offline.

E7 works around this for its own measurement by giving each rebuilt meeting one gain, computed over the whole meeting. A live
system cannot do that, so the workaround is a normalization of test audio and not a product feature. The product needs a gate
that adapts to the input level using only audio it has already heard (for example a running noise-floor estimate, or a running
gain control ahead of the gate), and a test that streams the same speech at several input levels and checks the words kept.
