// Microphone capture: 16 kHz mono 16-bit PCM chunks, base64-encoded.
export type StopMic = () => void;

export async function startMic(
  onChunk: (base64: string, sampleRate: number) => void,
  onEnded: () => void,
): Promise<StopMic> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  stream.getTracks().forEach((t) => t.addEventListener('ended', onEnded));
  const ctx = new AudioContext({ sampleRate: 16000 });
  const source = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(4096, 1, 1);
  node.onaudioprocess = (e) => {
    const f = e.inputBuffer.getChannelData(0);
    const pcm = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) pcm[i] = Math.max(-1, Math.min(1, f[i])) * 0x7fff;
    let bin = '';
    new Uint8Array(pcm.buffer).forEach((b) => {
      bin += String.fromCharCode(b);
    });
    onChunk(btoa(bin), ctx.sampleRate);
  };
  source.connect(node);
  node.connect(ctx.destination);
  return () => {
    node.disconnect();
    source.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    void ctx.close();
  };
}
