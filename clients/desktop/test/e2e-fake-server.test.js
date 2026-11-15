import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Controller } from '../src/main/controller.js';
import { DEFAULTS } from '../src/core/settings.js';
import { startFakeServer } from './helpers.js';

// The real controller, session and protocol code over a real WebSocket to tools/fake_server.py (scripted content, no model).
let fake;
beforeAll(async () => {
  fake = await startFakeServer();
}, 30000);
afterAll(() => fake?.stop());

const until = async (fn, ms = 5000) => {
  for (let i = 0; i < ms / 25; i++) {
    if (fn()) return;
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error('timed out');
};

function make() {
  const events = [];
  const c = new Controller({
    settings: { ...DEFAULTS, serverUrl: fake.url },
    emit: (ch, p) => events.push([ch, p]),
    health: async () => [],
    backend: { running: false, log: [], onChange() {}, stop() {} },
  });
  return { c, events };
}

describe('desktop client against the fake server', () => {
  it('connects, hears turns, presses three times (answered, partial, abstained); then forgets', async () => {
    const { c, events } = make();
    c.connect();
    await until(() => c.snapshot().connection === 'recording');
    expect(c.snapshot().capture.active).toBe(true);

    c.press();
    expect(events.some(([n, p]) => n === 'notice' && /Nothing heard/.test(p.text))).toBe(true);

    for (let i = 0; i < 5; i++) c.sendAudio('AAAA', 16000);
    await until(() => c.snapshot().turns.length === 5);
    await until(() => c.snapshot().turns[4].speaker !== 'unknown'); // from the fifth turn the speaker arrives as a turn_update
    expect(c.snapshot().turns[0]).toMatchObject({ speaker: 'Ana', seq: 0 });

    for (const status of ['answered', 'partial', 'abstained']) {
      const before = c.snapshot().cards.length;
      expect(c.press()).toBe(true);
      await until(() => c.snapshot().cards.length === before + 1);
      expect(events.filter(([n]) => n === 'card').at(-1)[1].card.status).toBe(status);
    }
    const answered = c.snapshot().cards[0].card;
    expect(answered.findings[0].evidence[0]).toMatchObject({ speaker: expect.any(String), quote: expect.any(String) });

    const old = c.settings.conversationId;
    c.forget();
    await until(() => c.settings.conversationId !== old);
    await until(() => c.snapshot().connection === 'recording');
    expect(c.snapshot().turns).toEqual([]);
    expect(c.snapshot().cards).toEqual([]);
    c.shutdown();
  }, 30000);

  it('paused audio produces no turns; a typed question is accepted', async () => {
    const { c } = make();
    c.connect();
    await until(() => c.snapshot().connection === 'recording');
    c.sendAudio('AAAA');
    await until(() => c.snapshot().turns.length === 1);
    c.pause();
    c.sendAudio('AAAA');
    await new Promise((r) => setTimeout(r, 300));
    expect(c.snapshot().turns).toHaveLength(1);
    expect(c.press('what was decided?')).toBe(true);
    await until(() => c.snapshot().cards.length === 1);
    expect(c.snapshot().cards[0].question).toBe('what was decided?');
    c.shutdown();
  }, 30000);
});
