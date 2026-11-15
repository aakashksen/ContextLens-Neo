import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { startFakeServer, stopTree } from './helpers.js';

// Launches the real Electron app (windows hidden, own temp user-data folder, no tray, no hotkeys, no microphone) against
// tools/fake_server.py. The app feeds fake audio, runs the press path and checks that the overlay and main window rendered.
const electronPath = createRequire(import.meta.url)('electron');
let fake;
beforeAll(async () => {
  fake = await startFakeServer();
}, 30000);
afterAll(() => fake?.stop());

describe('electron app smoke', () => {
  it('connects, shows the transcript, and renders a card in the overlay', async () => {
    const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'cl-desktop-'));
    const env = { ...process.env, CONTEXTLENS_DESKTOP_SMOKE: '1', CONTEXTLENS_SMOKE_URL: fake.url, CONTEXTLENS_USER_DATA: userData };
    delete env.ELECTRON_RUN_AS_NODE;
    const app = spawn(electronPath, ['.'], { cwd: path.resolve(import.meta.dirname, '..'), env, windowsHide: true });
    let out = '';
    app.stdout.on('data', (b) => (out += b));
    app.stderr.on('data', (b) => (out += b));
    const code = await new Promise((resolve) => {
      const t = setTimeout(() => (stopTree(app), resolve('timeout')), 50000);
      app.on('exit', (c) => (clearTimeout(t), resolve(c)));
    });
    fs.rmSync(userData, { recursive: true, force: true });
    const line = out.split('\n').find((l) => l.startsWith('SMOKE '));
    expect(line, out.slice(-800)).toBeTruthy();
    expect(JSON.parse(line.slice(6))).toEqual({ connected: true, transcript: true, cardInOverlay: true, mainWindowTurns: true });
    expect(code).toBe(0);
  }, 60000);
});
