import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

const finding = {
  agent: 'recall',
  kind: 'recall' as const,
  statement: 'Ana said the launch moves to Friday.',
  evidence: [{ turn_id: 'c-t0', quote: 'Let us move the launch to Friday.', speaker: 'Ana', start_ms: 65000 }],
  confidence: 0.9,
};

const turn = {
  id: 'c-t0',
  conversation_id: 'c',
  seq: 0,
  speaker: 'Ana',
  text: 'Let us move the launch to Friday.',
  start_ms: 0,
  end_ms: 3500,
  source: 'synthetic',
};

// A server that never answers pause or delete: the test pushes only what it chooses to.
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
  drop() {
    act(() => this.onclose?.());
  }
}

// Stands in for the browser microphone; the real getUserMedia path needs a browser.
class MicStub {
  stops = 0;
  onChunk: (b64: string, rate: number) => void = () => {};
  onEnded: () => void = () => {};
  private resolve: (stop: () => void) => void = () => {};
  private reject: (e: Error) => void = () => {};
  start = (onChunk: (b64: string, rate: number) => void, onEnded: () => void) => {
    this.onChunk = onChunk;
    this.onEnded = onEnded;
    return new Promise<() => void>((res, rej) => {
      this.resolve = res;
      this.reject = rej;
    });
  };
  async grant() {
    await act(async () => this.resolve(() => void this.stops++));
  }
  async deny() {
    await act(async () => this.reject(new Error('NotAllowedError')));
  }
}

function renderApp(mic: MicStub) {
  render(<App socketCtor={FakeSocket as any} startMic={mic.start} serverUrl="ws://test" conversationId="c" />);
  return FakeSocket.last;
}

const indicator = () => screen.getByRole('status', { name: /recording indicator/i });
const audioSent = (ws: FakeSocket) => ws.sent.filter((m) => m.type === 'audio').length;

async function listening(mic: MicStub) {
  const ws = renderApp(mic);
  ws.push({ type: 'status', state: 'recording' });
  await mic.grant();
  return ws;
}

async function confirmDelete() {
  await userEvent.click(screen.getByRole('button', { name: /delete/i }));
  await userEvent.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: /yes, delete/i }));
}

describe('recording indicator is honest', () => {
  it('says Recording only when audio is captured and the server is listening', async () => {
    const mic = new MicStub();
    const ws = renderApp(mic);
    expect(indicator()).toHaveTextContent(/connecting/i);
    expect(indicator()).not.toHaveTextContent(/^recording$/i);
    ws.push({ type: 'status', state: 'recording' });
    expect(indicator()).toHaveTextContent(/starting microphone/i);
    expect(indicator()).not.toHaveTextContent(/^recording$/i);
    await mic.grant();
    expect(indicator()).toHaveTextContent(/^recording$/i);
  });

  it('does not say Recording when the microphone is denied', async () => {
    const mic = new MicStub();
    const ws = renderApp(mic);
    ws.push({ type: 'status', state: 'recording' });
    await mic.deny();
    expect(indicator()).toHaveTextContent(/microphone blocked/i);
    expect(indicator()).not.toHaveTextContent(/^recording$/i);
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });

  it('does not say Recording after the microphone stops', async () => {
    const mic = new MicStub();
    await listening(mic);
    act(() => mic.onEnded());
    expect(indicator()).toHaveTextContent(/microphone stopped/i);
    expect(indicator()).not.toHaveTextContent(/^recording$/i);
  });

  it('does not say Recording when the server reports an error', async () => {
    const ws = await listening(new MicStub());
    ws.push({ type: 'status', state: 'error', detail: 'boom' });
    expect(indicator()).toHaveTextContent(/^error$/i);
  });

  it('says Paused and Deleted honestly', async () => {
    const ws = await listening(new MicStub());
    ws.push({ type: 'status', state: 'paused' });
    expect(indicator()).toHaveTextContent(/^paused$/i);
    ws.push({ type: 'status', state: 'deleted' });
    expect(indicator()).toHaveTextContent(/^deleted$/i);
  });

  it('says Disconnected when the socket closes', async () => {
    const ws = await listening(new MicStub());
    ws.drop();
    expect(indicator()).toHaveTextContent(/^disconnected$/i);
  });

  it('says Deleting until the server confirms', async () => {
    const ws = await listening(new MicStub());
    ws.push({ type: 'transcript', turn });
    await confirmDelete();
    expect(indicator()).toHaveTextContent(/^deleting$/i);
  });
});

describe('client-side stop', () => {
  it('pause stops the mic and audio at once, even if the server never answers', async () => {
    const mic = new MicStub();
    const ws = await listening(mic);
    mic.onChunk('AAA', 16000);
    expect(audioSent(ws)).toBe(1);
    await userEvent.click(screen.getByRole('button', { name: /pause/i }));
    expect(ws.sent.some((m) => m.type === 'pause')).toBe(true);
    expect(mic.stops).toBe(1);
    mic.onChunk('BBB', 16000);
    expect(audioSent(ws)).toBe(1);
    expect(indicator()).toHaveTextContent(/^paused$/i);
  });

  it('sends no audio after delete is confirmed, even if the server never answers', async () => {
    const mic = new MicStub();
    const ws = await listening(mic);
    ws.push({ type: 'transcript', turn });
    await confirmDelete();
    expect(mic.stops).toBe(1);
    mic.onChunk('CCC', 16000);
    expect(audioSent(ws)).toBe(0);
  });

  it('sends no audio after the socket closes', async () => {
    const mic = new MicStub();
    const ws = await listening(mic);
    ws.drop();
    expect(mic.stops).toBe(1);
    mic.onChunk('DDD', 16000);
    expect(audioSent(ws)).toBe(0);
  });

  it('a late recording status does not restart capture while deleting', async () => {
    const ws = await listening(new MicStub());
    ws.push({ type: 'transcript', turn });
    await confirmDelete();
    ws.push({ type: 'status', state: 'recording' });
    expect(indicator()).not.toHaveTextContent(/^recording$/i);
  });
});

describe('keyboard', () => {
  async function ready() {
    const ws = await listening(new MicStub());
    ws.push({ type: 'transcript', turn });
    ws.push({ type: 'card', card: { mode: 'recall', status: 'answered', findings: [finding] } });
    return { ws, user: userEvent.setup() };
  }

  it('every control can be reached with Tab', async () => {
    const { user } = await ready();
    const seen = new Set<HTMLElement>();
    for (let i = 0; i < 12; i++) {
      await user.tab();
      seen.add(document.activeElement as HTMLElement);
    }
    for (const el of [
      screen.getByRole('button', { name: /what was said/i }),
      screen.getByRole('button', { name: /pause/i }),
      screen.getByRole('button', { name: /delete/i }),
      screen.getByRole('button', { name: /show source/i }),
      screen.getByLabelText(/detail/i),
      screen.getByLabelText(/text size/i),
      screen.getByLabelText(/plain wording/i),
    ]) {
      expect(seen.has(el)).toBe(true);
    }
  });

  it('main button, pause and resume work with Enter and Space', async () => {
    const { ws, user } = await ready();
    screen.getByRole('button', { name: /what was said/i }).focus();
    await user.keyboard('{Enter}');
    expect(ws.sent.filter((m) => m.type === 'press')).toHaveLength(1);
    screen.getByRole('button', { name: /pause/i }).focus();
    await user.keyboard('{Enter}');
    expect(ws.sent.some((m) => m.type === 'pause')).toBe(true);
    screen.getByRole('button', { name: /resume/i }).focus();
    await user.keyboard(' ');
    expect(ws.sent.some((m) => m.type === 'resume')).toBe(true);
  });

  it('source toggle, selects and checkbox work from the keyboard', async () => {
    const { user } = await ready();
    const toggle = screen.getByRole('button', { name: /show source/i });
    toggle.focus();
    await user.keyboard('{Enter}');
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    await user.keyboard(' ');
    expect(toggle).toHaveAttribute('aria-expanded', 'false');

    const box = screen.getByLabelText(/plain wording/i);
    box.focus();
    await user.keyboard(' ');
    expect(box).toBeChecked();

    const size = screen.getByLabelText(/text size/i);
    size.focus();
    await user.selectOptions(size, 'large');
    expect(document.querySelector('.app')).toHaveClass('size-large');
    const detail = screen.getByLabelText(/detail/i) as HTMLSelectElement;
    detail.focus();
    await user.selectOptions(detail, 'detailed');
    expect(detail.value).toBe('detailed');
  });

  it('delete dialog: Escape and Cancel return focus; Yes deletes and focus stays on a control', async () => {
    const { ws, user } = await ready();
    const del = screen.getByRole('button', { name: /delete conversation/i });
    del.focus();
    await user.keyboard('{Enter}');
    expect(within(screen.getByRole('alertdialog')).getByRole('button', { name: /cancel/i })).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('alertdialog')).toBeNull();
    expect(del).toHaveFocus();

    await user.keyboard('{Enter}');
    await user.keyboard('{Enter}'); // Cancel has focus first
    expect(screen.queryByRole('alertdialog')).toBeNull();
    expect(del).toHaveFocus();
    expect(ws.sent.some((m) => m.type === 'delete')).toBe(false);

    await user.keyboard('{Enter}');
    await user.tab();
    expect(within(screen.getByRole('alertdialog')).getByRole('button', { name: /yes, delete/i })).toHaveFocus();
    await user.keyboard('{Enter}');
    expect(ws.sent.filter((m) => m.type === 'delete')).toHaveLength(1);
    expect(screen.queryByRole('alertdialog')).toBeNull();
    expect(document.body).not.toHaveFocus();
  });
});
