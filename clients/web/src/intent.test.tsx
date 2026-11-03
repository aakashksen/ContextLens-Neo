import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

// Capture and the Recording label may return only from a state where the user wants recording.

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
  status(state: string) {
    act(() => this.onmessage?.({ data: JSON.stringify({ type: 'status', state }) }));
  }
  drop() {
    act(() => this.onclose?.());
  }
  count(type: string) {
    return this.sent.filter((m) => m.type === type).length;
  }
}

// Every start is granted at once; counts starts and stops.
class MicStub {
  starts = 0;
  stops = 0;
  onChunk: (b64: string, rate: number) => void = () => {};
  start = (onChunk: (b64: string, rate: number) => void) => {
    this.starts++;
    this.onChunk = onChunk;
    return Promise.resolve(() => void this.stops++);
  };
}

const indicator = () => screen.getByRole('status', { name: /recording indicator/i });
const label = (re: RegExp) => expect(indicator()).toHaveTextContent(re);
const settle = () => act(async () => {});

async function setup(initial: 'recording' | null = 'recording') {
  const mic = new MicStub();
  render(<App socketCtor={FakeSocket as any} startMic={mic.start} serverUrl="ws://test" conversationId="c" />);
  const ws = FakeSocket.last;
  if (initial) ws.status(initial);
  await settle();
  return { ws, mic };
}

const pause = () => userEvent.click(screen.getByRole('button', { name: /pause/i }));
const resume = () => userEvent.click(screen.getByRole('button', { name: /resume/i }));
async function confirmDelete() {
  await userEvent.click(screen.getByRole('button', { name: /delete/i }));
  await userEvent.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: /yes, delete/i }));
}

describe('after delete is confirmed', () => {
  it('a stray recording status changes nothing and the microphone stays off', async () => {
    const { ws, mic } = await setup();
    await confirmDelete();
    ws.status('deleted');
    ws.status('recording');
    await settle();
    label(/^deleted$/i);
    expect(mic.starts).toBe(1);
    mic.onChunk('X', 16000);
    expect(ws.count('audio')).toBe(0);
  });

  it('a stray paused status changes nothing', async () => {
    const { ws } = await setup();
    await confirmDelete();
    ws.status('deleted');
    ws.status('paused');
    label(/^deleted$/i);
  });
});

describe('after a local pause', () => {
  it('a late recording status does not bring back Recording or the microphone', async () => {
    const { ws, mic } = await setup();
    await pause();
    ws.status('recording');
    await settle();
    label(/^paused$/i);
    expect(mic.starts).toBe(1);
    expect(mic.stops).toBe(1);
    mic.onChunk('X', 16000);
    expect(ws.count('audio')).toBe(0);
  });

  it('the server confirming paused keeps it paused', async () => {
    const { ws, mic } = await setup();
    await pause();
    ws.status('paused');
    ws.status('recording');
    await settle();
    label(/^paused$/i);
    expect(mic.starts).toBe(1);
  });

  it('pause, late recording, Resume, late paused, then recording resumes capture', async () => {
    const { ws, mic } = await setup();
    await pause();
    ws.status('recording');
    await resume();
    expect(ws.count('resume')).toBe(1);
    label(/^paused$/i);
    expect(mic.starts).toBe(1);
    ws.status('paused');
    label(/^paused$/i);
    expect(mic.starts).toBe(1);
    ws.status('recording');
    await settle();
    label(/^recording$/i);
    expect(mic.starts).toBe(2);
  });

  it('Resume pressed but not yet confirmed does not claim Recording or capture', async () => {
    const { mic } = await setup();
    await pause();
    await resume();
    await settle();
    label(/^paused$/i);
    expect(mic.starts).toBe(1);
  });

  it('a second Resume press sends one resume, and one confirmation starts one capture', async () => {
    const { ws, mic } = await setup();
    await pause();
    await resume();
    act(() => screen.getByRole('button', { name: /resume/i }).click());
    expect(ws.count('resume')).toBe(1);
    ws.status('recording');
    ws.status('recording');
    await settle();
    label(/^recording$/i);
    expect(mic.starts).toBe(2);
  });

  it('after a confirmed resume, a later pause and stray recording stay paused', async () => {
    const { ws, mic } = await setup();
    await pause();
    await resume();
    ws.status('recording');
    await settle();
    await pause();
    ws.status('recording');
    await settle();
    label(/^paused$/i);
    expect(mic.starts).toBe(2);
    expect(mic.stops).toBe(2);
  });
});

describe('a server-initiated pause', () => {
  it('stays paused on a stray recording and resumes only after the user presses Resume', async () => {
    const { ws, mic } = await setup();
    ws.status('paused');
    await settle();
    label(/^paused$/i);
    expect(mic.stops).toBe(1);
    ws.status('recording');
    await settle();
    label(/^paused$/i);
    expect(mic.starts).toBe(1);
    await resume();
    ws.status('recording');
    await settle();
    label(/^recording$/i);
    expect(mic.starts).toBe(2);
  });
});

describe('delete while paused', () => {
  it('shows Deleting, ignores a stray recording, then Deleted', async () => {
    const { ws, mic } = await setup();
    await pause();
    await confirmDelete();
    label(/^deleting$/i);
    ws.status('recording');
    await settle();
    label(/^deleting$/i);
    ws.status('deleted');
    ws.status('recording');
    await settle();
    label(/^deleted$/i);
    expect(mic.starts).toBe(1);
  });

  it('delete while a Resume is pending ignores the confirmation', async () => {
    const { ws, mic } = await setup();
    await pause();
    await resume();
    await confirmDelete();
    ws.status('recording');
    await settle();
    label(/^deleting$/i);
    expect(mic.starts).toBe(1);
  });
});

describe('socket close', () => {
  const afterClose = async (ws: FakeSocket, mic: MicStub, starts: number) => {
    label(/^disconnected$/i);
    ws.status('recording');
    ws.status('paused');
    await settle();
    label(/^disconnected$/i);
    expect(mic.starts).toBe(starts);
    mic.onChunk('X', 16000);
    expect(ws.count('audio')).toBe(0);
  };

  it('while connecting', async () => {
    const { ws, mic } = await setup(null);
    ws.drop();
    await afterClose(ws, mic, 0);
  });

  it('while recording', async () => {
    const { ws, mic } = await setup();
    ws.drop();
    await settle();
    await afterClose(ws, mic, 1);
  });

  it('while paused', async () => {
    const { ws, mic } = await setup();
    await pause();
    ws.drop();
    await afterClose(ws, mic, 1);
  });

  it('while a Resume is pending', async () => {
    const { ws, mic } = await setup();
    await pause();
    await resume();
    ws.drop();
    await afterClose(ws, mic, 1);
  });

  it('while deleting', async () => {
    const { ws, mic } = await setup();
    await confirmDelete();
    ws.drop();
    await afterClose(ws, mic, 1);
  });

  it('after deleted, the close keeps Deleted', async () => {
    const { ws, mic } = await setup();
    await confirmDelete();
    ws.status('deleted');
    ws.drop();
    ws.status('recording');
    await settle();
    label(/^deleted$/i);
    expect(mic.starts).toBe(1);
  });
});
