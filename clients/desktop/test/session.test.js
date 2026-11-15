import { beforeEach, describe, expect, it } from 'vitest';
import { Session } from '../src/core/session.js';
import { floatToBase64Pcm } from '../src/core/audio.js';
import { FakeSocket, card, turn } from './helpers.js';

let sess, sock;
beforeEach(() => {
  sess = new Session('ws://x/ws/c', 'c', { Socket: FakeSocket });
  sock = FakeSocket.last;
  sock.open();
});
const recording = () => sock.push({ type: 'status', state: 'recording' });

describe('Session', () => {
  it('sends audio only while recording', () => {
    sess.sendAudio('AA');
    expect(sock.sent).toEqual([]);
    recording();
    sess.sendAudio('AA', 16000);
    expect(sock.sent).toEqual([{ type: 'audio', data: 'AA', sample_rate: 16000 }]);
  });

  it('appends transcripts and replaces a turn on turn_update, ignoring unknown ids', () => {
    sock.push({ type: 'transcript', turn: turn(0, 'unknown') });
    sock.push({ type: 'turn_update', turn: turn(0, 'Ana') });
    sock.push({ type: 'turn_update', turn: { ...turn(9), id: 'other' } });
    expect(sess.state.turns.map((t) => t.speaker)).toEqual(['Ana']);
  });

  it('presses with the latest seq, and refuses with nothing heard', () => {
    recording();
    expect(sess.press('standard')).toBe(false);
    sock.push({ type: 'transcript', turn: turn(0) });
    sock.push({ type: 'transcript', turn: turn(1) });
    expect(sess.press('brief', 'when?')).toBe(true);
    expect(sock.sent.at(-1)).toEqual({ type: 'press', request: { conversation_id: 'c', as_of_seq: 1, detail: 'brief', question: 'when?' } });
    expect(sess.state.waiting).toBe(true);
    sock.push({ type: 'card', card: card() });
    expect(sess.state.waiting).toBe(false);
    expect(sess.state.cards).toHaveLength(1);
    expect(sess.state.cards[0].question).toBe('when?');
  });

  it('pause stops audio and a stale recording status does not resume it', () => {
    recording();
    sess.pause();
    recording();
    expect(sess.state.connection).toBe('paused');
    sess.sendAudio('AA');
    expect(sock.sent.filter((m) => m.type === 'audio')).toEqual([]);
    sess.resume();
    recording();
    expect(sess.state.connection).toBe('recording');
  });

  it('after an error while recording, audio stays off', () => {
    recording();
    sock.push({ type: 'status', state: 'error', detail: 'boom' });
    expect(sess.state.error).toBe('boom');
    sess.sendAudio('AA');
    expect(sock.sent).toEqual([]);
  });

  it('delete clears everything and drops late messages', () => {
    recording();
    sock.push({ type: 'transcript', turn: turn(0) });
    sess.delete();
    expect(sock.sent.at(-1)).toEqual({ type: 'delete' });
    sock.push({ type: 'status', state: 'deleted' });
    sock.push({ type: 'transcript', turn: turn(1) });
    sock.push({ type: 'card', card: card() });
    expect(sess.state).toMatchObject({ connection: 'deleted', turns: [], cards: [], card: null });
  });

  it('a close ends the session', () => {
    recording();
    sock.drop();
    expect(sess.state.connection).toBe('closed');
  });
});

describe('floatToBase64Pcm', () => {
  it('encodes 16-bit little-endian PCM and clamps', () => {
    const b64 = floatToBase64Pcm(new Float32Array([0, 1, -1, 2]));
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const pcm = new Int16Array(bytes.buffer);
    expect(Array.from(pcm)).toEqual([0, 32767, -32767, 32767]);
  });
});
