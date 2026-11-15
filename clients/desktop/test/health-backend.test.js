import { EventEmitter } from 'node:events';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { BackendProcess, resolveBackend } from '../src/core/backend.js';
import { checkAll } from '../src/core/health.js';

const json = (body, status = 200) => Promise.resolve({ status, json: async () => body });
const byId = (r) => Object.fromEntries(r.map((x) => [x.id, x]));

describe('checkAll', () => {
  const base = { serverUrl: 'ws://127.0.0.1:8000' };
  it('all ready', async () => {
    const fetchFn = (u) => (u.endsWith('/api/tags') ? json({ models: [{ name: 'qwen3.5:4b-q4_K_M' }, { name: 'nomic-embed-text:latest' }] }) : json({}));
    const r = byId(await checkAll({ ...base, fetchFn, tcp: async () => true }));
    expect(Object.values(r).map((x) => x.state)).toEqual(['ready', 'ready', 'ready', 'ready']);
  });
  it('everything down gives one fix command each', async () => {
    const r = byId(await checkAll({ ...base, fetchFn: () => Promise.reject(new Error('x')), tcp: async () => false }));
    expect(r.ollama).toMatchObject({ state: 'missing', fix: 'ollama serve' });
    expect(r.speech.fix).toBe('docker compose up -d speech');
    expect(r.db.fix).toBe('docker compose up -d db');
    expect(r.api.fix).toContain('Start backend');
  });
  it('names the missing model and reports a warming API', async () => {
    const fetchFn = (u) => (u.endsWith('/api/tags') ? json({ models: [{ name: 'nomic-embed-text' }] }) : json({}, 503));
    const r = byId(await checkAll({ ...base, fetchFn, tcp: async () => true }));
    expect(r.ollama.fix).toBe('ollama pull qwen3.5:4b-q4_K_M');
    expect(r.api.state).toBe('warming');
  });
});

describe('resolveBackend', () => {
  const have = (set) => (p) => set.includes(path.normalize(p));
  it('finds the repo and its venv python', () => {
    const root = path.resolve('/repo');
    const py = path.join(root, '.venv', 'Scripts', 'python.exe');
    expect(resolveBackend({ devRoot: root, exists: have([path.join(root, 'src', 'contextlens'), py]), platform: 'win32' })).toEqual({ root, python: py });
  });
  it('explains what is missing', () => {
    expect(resolveBackend({ exists: () => false }).error).toContain('repository not found');
    const root = path.resolve('/repo');
    expect(resolveBackend({ repoRoot: root, exists: have([path.join(root, 'src', 'contextlens')]) }).error).toContain('Python not found');
  });
});

describe('BackendProcess', () => {
  const fakeChild = () => Object.assign(new EventEmitter(), { stdout: new EventEmitter(), stderr: new EventEmitter(), pid: 7 });
  it('starts the production entry with the chosen assistant and stops what it started', () => {
    const child = fakeChild();
    const spawnFn = vi.fn(() => child);
    const kill = vi.fn();
    const b = new BackendProcess({ spawnFn, kill });
    b.start({ root: '/r', python: '/r/py', assistant: 'single_plus' });
    expect(spawnFn.mock.calls[0][0]).toBe('/r/py');
    expect(spawnFn.mock.calls[0][1]).toEqual(['-m', 'contextlens.api.production']);
    expect(spawnFn.mock.calls[0][2].env.CONTEXTLENS_ASSISTANT).toBe('single_plus');
    expect(b.running).toBe(true);
    child.stdout.emit('data', 'warming\n');
    expect(b.log).toEqual(['warming']);
    b.start({ root: '/r', python: '/r/py', assistant: 'team' });
    expect(spawnFn).toHaveBeenCalledTimes(1);
    b.stop();
    expect(kill).toHaveBeenCalledWith(child);
    expect(b.running).toBe(false);
  });
  it('clears running when the process exits by itself', () => {
    const child = fakeChild();
    const b = new BackendProcess({ spawnFn: () => child, kill: () => {} });
    b.start({ root: '/r', python: '/p', assistant: 'team' });
    child.emit('exit', 1);
    expect(b.running).toBe(false);
    expect(b.log.at(-1)).toContain('exited');
  });
});
