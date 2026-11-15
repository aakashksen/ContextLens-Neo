// 16 kHz mono 16-bit PCM, base64: what clients/web sends.
export function floatToBase64Pcm(f32) {
  const pcm = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) pcm[i] = Math.max(-1, Math.min(1, f32[i])) * 0x7fff;
  const bytes = new Uint8Array(pcm.buffer);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}
