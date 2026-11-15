// Writes assets/icon.png (512), tray.png and tray-live.png (32): a lens ring, red dot while listening. No dependencies.
import { deflateSync } from 'node:zlib';
import fs from 'node:fs';

function crc32(buf) {
  let c, crc = ~0;
  for (const b of buf) {
    c = (crc ^ b) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = (crc >>> 8) ^ c;
  }
  return ~crc >>> 0;
}
const chunk = (type, data) => {
  const t = Buffer.from(type);
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([t, data])));
  return Buffer.concat([len, t, data, crc]);
};
function png(size, pixel) {
  const raw = Buffer.alloc((size * 4 + 1) * size);
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0;
    for (let x = 0; x < size; x++) {
      const [r, g, b, a] = pixel((x + 0.5) / size, (y + 0.5) / size);
      raw.set([r, g, b, a], y * (size * 4 + 1) + 1 + x * 4);
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr.set([8, 6, 0, 0, 0], 8);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}
const lens = (live, background) => (u, v) => {
  const d = Math.hypot(u - 0.5, v - 0.5);
  if (live && Math.hypot(u - 0.5, v - 0.5) < 0.17) return [255, 90, 90, 255];
  if (d > 0.22 && d < 0.4) return [106, 169, 255, 255];
  if (background && d < 0.48) return [20, 22, 26, 255];
  return [0, 0, 0, 0];
};
fs.mkdirSync('assets', { recursive: true });
fs.writeFileSync('assets/icon.png', png(512, lens(false, true)));
fs.writeFileSync('assets/tray.png', png(32, lens(false, false)));
fs.writeFileSync('assets/tray-live.png', png(32, lens(true, false)));
