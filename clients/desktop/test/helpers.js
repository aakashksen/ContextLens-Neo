import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
export const repoRoot = path.resolve(here, '..', '..', '..');

export class FakeSocket {
  static last = null;
  readyState = 0;
  sent = [];
  constructor(url) {
    this.url = url;
    FakeSocket.last = this;
  }
  send(s) {
    this.sent.push(JSON.parse(s));
  }
  close() {
    this.readyState = 3;
  }
  open() {
    this.readyState = 1;
  }
  push(msg) {
    this.onmessage?.({ data: typeof msg === 'string' ? msg : JSON.stringify(msg) });
  }
  drop() {
    this.readyState = 3;
    this.onclose?.();
  }
}

export const turn = (seq, speaker = 'Ana', text = `turn ${seq}`) => ({
  id: `c-t${seq}`,
  conversation_id: 'c',
  seq,
  speaker,
  text,
  start_ms: seq * 4000,
  end_ms: seq * 4000 + 3500,
  source: 'synthetic',
});
export const evidence = (t) => ({ turn_id: t.id, quote: t.text, speaker: t.speaker, start_ms: t.start_ms });
export const card = (status = 'answered', extra = {}) => ({
  mode: 'recall',
  status,
  findings: status === 'abstained' ? [] : [{ agent: 'recall', kind: 'recall', statement: 'Ana said Friday.', evidence: [evidence(turn(0))], confidence: 0.9 }],
  ...extra,
});

export function pythonForTests() {
  const candidates = [
    process.env.CONTEXTLENS_PYTHON,
    path.join(repoRoot, '.venv', 'Scripts', 'python.exe'),
    path.join(repoRoot, '..', '..', '..', 'ContextLensV2', '.venv', 'Scripts', 'python.exe'),
    path.join(repoRoot, '.venv', 'bin', 'python'),
  ].filter(Boolean);
  return candidates.find((p) => fs.existsSync(p));
}

function freePort() {
  return new Promise((resolve) => {
    const s = net.createServer().listen(0, '127.0.0.1', () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });
}

export function stopTree(child) {
  if (process.platform === 'win32') spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
  else child.kill();
}

// Starts tools/fake_server.py (scripted content, no model) on a free port. Never port 8000.
export async function startFakeServer() {
  const python = pythonForTests();
  if (!python) throw new Error('no python with the contextlens server extras found; set CONTEXTLENS_PYTHON');
  const port = await freePort();
  const code = `import sys, uvicorn; sys.path.insert(0, 'tools'); from fake_server import create_app; uvicorn.run(create_app(), host='127.0.0.1', port=${port}, log_level='warning')`;
  const child = spawn(python, ['-c', code], { cwd: repoRoot, windowsHide: true, stdio: 'ignore' });
  for (let i = 0; i < 100; i++) {
    const up = await new Promise((resolve) => {
      const s = net.connect({ host: '127.0.0.1', port });
      s.once('connect', () => {
        s.destroy();
        resolve(true);
      });
      s.once('error', () => resolve(false));
    });
    if (up) return { url: `ws://127.0.0.1:${port}`, stop: () => stopTree(child) };
    await new Promise((r) => setTimeout(r, 200));
  }
  stopTree(child);
  throw new Error('fake server did not start');
}
