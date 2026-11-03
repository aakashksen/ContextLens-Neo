import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import Ajv from 'ajv';
import { Session } from './session';

const schemaPath = resolve(process.cwd(), '../../docs/client-contract.schema.json');
const schema = JSON.parse(readFileSync(schemaPath, 'utf8'));

class FakeSocket {
  readyState = 1;
  sent: string[] = [];
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  send(s: string) {
    this.sent.push(s);
  }
  close() {}
  push(msg: object) {
    this.onmessage?.({ data: JSON.stringify(msg) });
  }
}

const turn = {
  id: 'c-t0',
  conversation_id: 'c',
  seq: 7,
  speaker: 'Ana',
  text: 'hi',
  start_ms: 0,
  end_ms: 1,
  source: 'synthetic',
};

describe('outgoing frames match docs/client-contract.schema.json', () => {
  it('every frame the client can send validates', () => {
    const ajv = new Ajv({ strict: false });
    const validate = ajv.compile(schema.client_to_server);
    let socket!: FakeSocket;
    const Ctor = class extends FakeSocket {
      constructor(_url: string) {
        super();
        socket = this;
      }
    };
    const s = new Session('ws://x', 'c', Ctor as any);
    socket.push({ type: 'status', state: 'recording' });
    socket.push({ type: 'transcript', turn });
    s.sendAudio('AAAA');
    s.press('brief');
    s.pause();
    s.resume();
    s.delete();

    const types = socket.sent.map((f) => JSON.parse(f).type);
    expect(types).toEqual(['audio', 'press', 'pause', 'resume', 'delete']);
    for (const frame of socket.sent) {
      expect(validate(JSON.parse(frame)), `${frame}: ${ajv.errorsText(validate.errors)}`).toBe(true);
    }
  });

  it('the validator rejects a malformed frame', () => {
    const validate = new Ajv({ strict: false }).compile(schema.client_to_server);
    expect(validate({ type: 'press', request: { conversation_id: 'c' } })).toBe(false);
  });
});
