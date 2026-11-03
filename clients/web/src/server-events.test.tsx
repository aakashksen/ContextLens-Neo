import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import { parseServerMessage } from './protocol';

const turn = {
  id: 'c-t0',
  conversation_id: 'c',
  seq: 0,
  speaker: 'unknown',
  speaker_confidence: 0,
  text: 'Let us move the launch to Friday.',
  start_ms: 0,
  end_ms: 3500,
  source: 'asr',
};

const card = { mode: 'recall', status: 'abstained', findings: [], missing: ['nothing matched'] };

class FakeSocket {
  readyState = 1;
  sent: any[] = [];
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  static last: FakeSocket;
  constructor(public url: string) {
    FakeSocket.last = this;
  }
  send(s: string) {
    this.sent.push(JSON.parse(s));
  }
  close() {}
  push(msg: object) {
    act(() => this.onmessage?.({ data: JSON.stringify(msg) }));
  }
}

class MicStub {
  stops = 0;
  starts = 0;
  onChunk: (b64: string, rate: number) => void = () => {};
  private resolve: (stop: () => void) => void = () => {};
  start = (onChunk: (b64: string, rate: number) => void) => {
    this.starts++;
    this.onChunk = onChunk;
    return new Promise<() => void>((res) => {
      this.resolve = res;
    });
  };
  async grant() {
    await act(async () => this.resolve(() => void this.stops++));
  }
}

async function listening(mic: MicStub) {
  render(<App socketCtor={FakeSocket as any} startMic={mic.start} serverUrl="ws://test" conversationId="c" />);
  const ws = FakeSocket.last;
  ws.push({ type: 'status', state: 'recording' });
  await mic.grant();
  return ws;
}

const indicator = () => screen.getByRole('status', { name: /recording indicator/i });
const audioSent = (ws: FakeSocket) => ws.sent.filter((m) => m.type === 'audio').length;

describe('a server error while recording', () => {
  it('stops the microphone and sends no more audio', async () => {
    const mic = new MicStub();
    const ws = await listening(mic);
    mic.onChunk('AAAA', 16000);
    expect(audioSent(ws)).toBe(1);
    ws.push({ type: 'status', state: 'error', detail: 'speech service unavailable' });
    expect(mic.stops).toBe(1);
    expect(indicator()).toHaveTextContent(/^error$/i);
    mic.onChunk('AAAA', 16000);
    expect(audioSent(ws)).toBe(1);
  });

  it('stays stopped when the user presses the button, and restarts after pause and resume', async () => {
    const mic = new MicStub();
    const ws = await listening(mic);
    ws.push({ type: 'transcript', turn });
    ws.push({ type: 'status', state: 'error', detail: 'boom' });
    await userEvent.click(screen.getByRole('button', { name: /what was said/i }));
    expect(mic.stops).toBe(1);
    expect(mic.starts).toBe(1);
    expect(indicator()).toHaveTextContent(/^error$/i);
    await userEvent.click(screen.getByRole('button', { name: /pause/i }));
    await userEvent.click(screen.getByRole('button', { name: /resume/i }));
    ws.push({ type: 'status', state: 'recording' });
    await mic.grant();
    expect(mic.starts).toBe(2);
    expect(indicator()).toHaveTextContent(/^recording$/i);
  });
});

describe('after the conversation is deleted', () => {
  it('ignores a late transcript, turn update and card', async () => {
    const ws = await listening(new MicStub());
    ws.push({ type: 'status', state: 'deleted' });
    ws.push({ type: 'transcript', turn });
    ws.push({ type: 'turn_update', turn });
    ws.push({ type: 'card', card });
    expect(screen.queryByText(/move the launch/i)).toBeNull();
    expect(screen.queryByText(/nothing matched/i)).toBeNull();
    expect(indicator()).toHaveTextContent(/^deleted$/i);
  });
});

describe('turn_update', () => {
  it('is accepted by the parser only with a turn', () => {
    expect(parseServerMessage(JSON.stringify({ type: 'turn_update', turn }))?.type).toBe('turn_update');
    expect(parseServerMessage(JSON.stringify({ type: 'turn_update' }))).toBeNull();
  });

  it('replaces the speaker of a turn already shown, without adding a second line', async () => {
    const ws = await listening(new MicStub());
    ws.push({ type: 'transcript', turn });
    expect(screen.getByText(/move the launch/i).closest('li')).toHaveTextContent(/^unknown/);
    ws.push({ type: 'turn_update', turn: { ...turn, speaker: 'speaker_1', speaker_confidence: 0.9 } });
    expect(screen.getAllByText(/move the launch/i)).toHaveLength(1);
    expect(screen.getByText(/move the launch/i).closest('li')).toHaveTextContent(/^speaker_1/);
  });

  it('is ignored for a turn the client never received', async () => {
    const ws = await listening(new MicStub());
    ws.push({ type: 'turn_update', turn });
    expect(screen.queryByText(/move the launch/i)).toBeNull();
  });
});
