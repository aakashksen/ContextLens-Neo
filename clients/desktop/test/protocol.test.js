import { describe, expect, it } from 'vitest';
import { parseServerMessage, pressMessage, audioMessage } from '../src/core/protocol.js';
import { card, turn } from './helpers.js';

const s = JSON.stringify;
describe('parseServerMessage', () => {
  it('accepts each contract message', () => {
    expect(parseServerMessage(s({ type: 'transcript', turn: turn(0) }))?.type).toBe('transcript');
    expect(parseServerMessage(s({ type: 'turn_update', turn: turn(0) }))?.type).toBe('turn_update');
    expect(parseServerMessage(s({ type: 'card', card: card() }))?.type).toBe('card');
    expect(parseServerMessage(s({ type: 'status', state: 'recording' }))?.type).toBe('status');
    expect(parseServerMessage(s({ type: 'memory', items: ['x'] }))?.type).toBe('memory');
  });
  it('rejects binary, junk, unknown types and malformed payloads', () => {
    const bad = [new ArrayBuffer(2), 'not json', s(null), s({ type: 'nope' }), s({ type: 'card', card: { status: 'weird', findings: [] } }), s({ type: 'status', state: 'asleep' }), s({ type: 'transcript', turn: { seq: 'x' } })];
    for (const b of bad) expect(parseServerMessage(b)).toBeNull();
  });
});

describe('client messages', () => {
  it('audio defaults to 16 kHz', () => expect(audioMessage('AA')).toEqual({ type: 'audio', data: 'AA', sample_rate: 16000 }));
  it('press carries the question only when given', () => {
    expect(pressMessage('c', 3, 'brief').request).toEqual({ conversation_id: 'c', as_of_seq: 3, detail: 'brief' });
    expect(pressMessage('c', 3, 'brief', 'when?').request.question).toBe('when?');
  });
});
