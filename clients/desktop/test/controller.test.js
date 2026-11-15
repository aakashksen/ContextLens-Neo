import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Controller } from '../src/main/controller.js';
import { DEFAULTS } from '../src/core/settings.js';
import { FakeSocket, card, turn } from './helpers.js';

let c, events, saved, backend;
beforeEach(() => {
  events = [];
  saved = [];
  backend = {
    running: false,
    log: [],
    start: vi.fn(function () {
      this.running = true;
    }),
    stop: vi.fn(function () {
      this.running = false;
    }),
    onChange: () => {},
  };
  c = new Controller({
    settings: { ...DEFAULTS },
    saveSettings: (s) => saved.push({ ...s }),
    emit: (ch, p) => events.push([ch, p]),
    backend,
    health: async () => [],
    Socket: FakeSocket,
    retryMs: 10,
    setTimer: (fn) => {
      fn();
      return 0;
    },
    clearTimer: () => {},
  });
  c.connect();
  FakeSocket.last.open();
  FakeSocket.last.push({ type: 'status', state: 'recording' });
});
const sock = () => FakeSocket.last;
const of = (name) => events.filter(([n]) => n === name).map(([, p]) => p);

describe('Controller', () => {
  it('connects to /ws/<conversation id> and is listening once recording', () => {
    expect(sock().url).toBe(`ws://127.0.0.1:8000/ws/${c.settings.conversationId}`);
    expect(c.snapshot().capture).toEqual({ active: true, mic: true, loopback: false });
  });

  it('press with nothing heard says so; with a turn it sends the configured detail and the question', () => {
    expect(c.press()).toBe(false);
    expect(of('notice').at(-1).text).toContain('Nothing heard');
    sock().push({ type: 'transcript', turn: turn(0) });
    c.setSettings({ detail: 'detailed' });
    expect(c.press('  when?  ')).toBe(true);
    expect(sock().sent.at(-1).request).toMatchObject({ as_of_seq: 0, detail: 'detailed', question: 'when?' });
    sock().push({ type: 'card', card: card() });
    expect(of('card')).toHaveLength(1);
    expect(of('card')[0]).toMatchObject({ question: 'when?', detail: 'detailed' });
    expect(c.snapshot().cards).toHaveLength(1);
  });

  it('pause stops capture and survives a reconnect', () => {
    c.pause();
    expect(c.snapshot().capture.active).toBe(false);
    sock().drop(); // the retry timer fires at once and opens a new socket
    FakeSocket.last.open();
    FakeSocket.last.push({ type: 'status', state: 'recording' });
    expect(FakeSocket.last.sent.at(-1)).toEqual({ type: 'pause' });
    expect(c.snapshot().capture.active).toBe(false);
    c.resume();
    FakeSocket.last.push({ type: 'status', state: 'recording' });
    expect(c.snapshot().capture.active).toBe(true);
  });

  it('keeps turns across a reconnect', () => {
    sock().push({ type: 'transcript', turn: turn(0) });
    sock().drop();
    expect(c.snapshot().turns).toHaveLength(1);
  });

  it('forget deletes on the server, clears local history and starts a new conversation', () => {
    sock().push({ type: 'transcript', turn: turn(0) });
    const old = c.settings.conversationId;
    c.forget();
    expect(sock().sent.at(-1)).toEqual({ type: 'delete' });
    const first = sock();
    first.push({ type: 'status', state: 'deleted' });
    expect(c.settings.conversationId).not.toBe(old);
    expect(saved.at(-1).conversationId).toBe(c.settings.conversationId);
    expect(c.snapshot().turns).toEqual([]);
    expect(FakeSocket.last).not.toBe(first);
    expect(FakeSocket.last.url).toContain(c.settings.conversationId);
  });

  it('settings: rejects bad values, saves good ones, reconnects on a new server', () => {
    const first = sock();
    expect(c.setSettings({ detail: 'nope', mic: false }).rejected).toEqual(['detail']);
    expect(c.settings.mic).toBe(false);
    expect(saved.at(-1).mic).toBe(false);
    c.setSettings({ serverUrl: 'ws://127.0.0.1:9999' });
    expect(FakeSocket.last).not.toBe(first);
    expect(FakeSocket.last.url).toContain('127.0.0.1:9999');
  });

  it('start backend: a missing repo is reported, not thrown', () => {
    c.settings.repoRoot = '/definitely/not/here';
    c.devRoot = '';
    expect(c.startBackend()).toBe(false);
    expect(c.snapshot().backend.error).toContain('repository not found');
    expect(backend.start).not.toHaveBeenCalled();
  });

  it('shutdown stops the backend it owns', () => {
    c.shutdown();
    expect(backend.stop).toHaveBeenCalled();
  });
});
