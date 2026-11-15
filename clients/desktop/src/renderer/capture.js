import { floatToBase64Pcm } from '../core/audio.js';

// Hidden window. Main tells it what to capture; it streams 16 kHz mono PCM chunks back, like clients/web does.
// Microphone and (Windows) system loopback are summed into one stream so both sides of a call are heard.
let running = null;

async function start({ mic, loopback }) {
  const streams = [];
  try {
    if (mic) streams.push(await navigator.mediaDevices.getUserMedia({ audio: true }));
    if (loopback) {
      const d = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
      d.getVideoTracks().forEach((t) => {
        t.stop();
        d.removeTrack(t);
      });
      if (d.getAudioTracks().length === 0) throw new Error('No system audio was provided.');
      streams.push(d);
    }
    if (!streams.length) throw new Error('No audio source is switched on.');
    const ctx = new AudioContext({ sampleRate: 16000 });
    const node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (e) => window.cl.send('audio-chunk', floatToBase64Pcm(e.inputBuffer.getChannelData(0)), ctx.sampleRate);
    const sources = streams.map((s) => ctx.createMediaStreamSource(s));
    sources.forEach((s) => s.connect(node));
    node.connect(ctx.destination);
    streams.forEach((s) => s.getTracks().forEach((t) => t.addEventListener('ended', () => window.cl.send('capture-error', 'An audio source stopped.'))));
    window.cl.send('capture-error', null);
    return () => {
      node.onaudioprocess = null;
      node.disconnect();
      sources.forEach((s) => s.disconnect());
      streams.forEach((s) => s.getTracks().forEach((t) => t.stop()));
      void ctx.close();
    };
  } catch (e) {
    streams.forEach((s) => s.getTracks().forEach((t) => t.stop()));
    throw e;
  }
}

let want = null;
let busy = false;
async function reconcile() {
  if (busy) return;
  busy = true;
  try {
    for (;;) {
      const w = want;
      const key = w?.active ? `${w.mic}|${w.loopback}` : '';
      if ((running?.key ?? '') === key) break;
      if (running) {
        running.stop();
        running = null;
      }
      if (key) {
        try {
          running = { key, stop: await start(w) };
        } catch (e) {
          window.cl.send('capture-error', `Could not capture audio: ${e.message}`);
          break;
        }
      }
    }
  } finally {
    busy = false;
  }
}

window.cl.on('capture', (c) => {
  want = c;
  void reconcile();
});
