import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import CardView from './components/CardView';
import type { Card } from './protocol';

const finding = {
  agent: 'recall',
  kind: 'recall' as const,
  statement: 'Ana said the launch moves to Friday.',
  evidence: [{ turn_id: 'c-t0', quote: 'Let us move the launch to Friday.', speaker: 'Ana', start_ms: 65000 }],
  confidence: 0.9,
};

class FakeSocket {
  static last: FakeSocket;
  readyState = 1;
  sent: any[] = [];
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
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

function renderApp() {
  render(
    <App
      socketCtor={FakeSocket as any}
      startMic={async () => () => {}}
      serverUrl="ws://test"
      conversationId="c"
    />,
  );
  return FakeSocket.last;
}

describe('card view', () => {
  const answered: Card = {
    mode: 'recall',
    status: 'answered',
    findings: [finding],
    suggestion: 'Ask Ana whether the sheet is ready.',
  };

  it('suggestion is labeled and styled apart from recalled facts', () => {
    render(<CardView card={answered} plain={false} />);
    const facts = screen.getByRole('region', { name: /from the conversation/i });
    const suggestion = screen.getByRole('region', { name: /suggestion/i });
    expect(facts).not.toBe(suggestion);
    expect(facts).toHaveClass('recalled');
    expect(suggestion).toHaveClass('suggestion');
    expect(suggestion).not.toHaveClass('recalled');
    expect(within(facts).getByText(finding.statement)).toBeInTheDocument();
    expect(within(facts).queryByText(/Ask Ana/)).toBeNull();
    expect(within(suggestion).getByText(/Ask Ana/)).toBeInTheDocument();
    expect(within(suggestion).getByText(/not from the conversation/i)).toBeInTheDocument();
  });

  it('abstained card says not enough information', () => {
    render(<CardView card={{ mode: 'recall', status: 'abstained', findings: [], missing: ['the due date'] }} plain />);
    expect(screen.getByText(/not enough information/i)).toBeInTheDocument();
    expect(screen.getByText('the due date')).toBeInTheDocument();
  });

  it('shows the source of a finding on request', async () => {
    render(<CardView card={answered} plain={false} />);
    expect(screen.queryByText(/Let us move the launch/)).toBeNull();
    await userEvent.click(screen.getByRole('button', { name: /show source/i }));
    expect(screen.getByText(/Let us move the launch/)).toBeInTheDocument();
    expect(screen.getByText(/Ana/, { selector: '.source-speaker' })).toBeInTheDocument();
    expect(screen.getByText('1:05')).toBeInTheDocument();
  });

  it('renders card text as text, never as html', () => {
    const evil = { ...finding, statement: '<img src=x onerror=alert(1)>' };
    const { container } = render(<CardView card={{ ...answered, findings: [evil] }} plain={false} />);
    expect(container.querySelector('img')).toBeNull();
    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeInTheDocument();
  });
});

describe('app', () => {
  it('recording indicator shows while listening', () => {
    const ws = renderApp();
    ws.push({ type: 'status', state: 'recording' });
    expect(screen.getByRole('status', { name: /recording/i })).toHaveTextContent(/recording/i);
    ws.push({ type: 'status', state: 'paused' });
    expect(screen.getByRole('status', { name: /recording/i })).toHaveTextContent(/paused/i);
  });

  it('delete asks to confirm', async () => {
    const ws = renderApp();
    ws.push({ type: 'status', state: 'recording' });
    ws.push({ type: 'transcript', turn });
    await userEvent.click(screen.getByRole('button', { name: /delete/i }));
    expect(ws.sent.some((m) => m.type === 'delete')).toBe(false);
    const dialog = screen.getByRole('alertdialog');
    await userEvent.click(within(dialog).getByRole('button', { name: /cancel/i }));
    expect(ws.sent.some((m) => m.type === 'delete')).toBe(false);
    await userEvent.click(screen.getByRole('button', { name: /delete/i }));
    await userEvent.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: /yes, delete/i }));
    expect(ws.sent.filter((m) => m.type === 'delete')).toHaveLength(1);
    ws.push({ type: 'status', state: 'deleted' });
    expect(screen.queryByText(turn.text)).toBeNull();
  });

  it('pressing the button sends the latest seq and detail, and shows the card', async () => {
    const ws = renderApp();
    ws.push({ type: 'status', state: 'recording' });
    const button = screen.getByRole('button', { name: /what was said/i });
    expect(button).toBeDisabled();
    ws.push({ type: 'transcript', turn });
    await userEvent.selectOptions(screen.getByLabelText(/detail/i), 'brief');
    await userEvent.click(button);
    const press = ws.sent.find((m) => m.type === 'press');
    expect(press.request).toMatchObject({ conversation_id: 'c', as_of_seq: 0, detail: 'brief' });
    ws.push({ type: 'card', card: { mode: 'recall', status: 'answered', findings: [finding] } });
    expect(screen.getByText(finding.statement)).toBeInTheDocument();
  });

  it('pause and resume send messages; keyboard activates controls', async () => {
    const ws = renderApp();
    ws.push({ type: 'status', state: 'recording' });
    const user = userEvent.setup();
    await user.tab();
    const first = document.activeElement as HTMLElement;
    expect(first).toBeInstanceOf(HTMLElement);
    screen.getByRole('button', { name: /pause/i }).focus();
    await user.keyboard('{Enter}');
    expect(ws.sent.some((m) => m.type === 'pause')).toBe(true);
    ws.push({ type: 'status', state: 'paused' });
    screen.getByRole('button', { name: /resume/i }).focus();
    await user.keyboard(' ');
    expect(ws.sent.some((m) => m.type === 'resume')).toBe(true);
  });

  it('text size setting changes the root size class', async () => {
    renderApp();
    await userEvent.selectOptions(screen.getByLabelText(/text size/i), 'large');
    expect(document.querySelector('.app')).toHaveClass('size-large');
  });

  it('plain wording setting changes label wording', async () => {
    const ws = renderApp();
    ws.push({ type: 'status', state: 'recording' });
    ws.push({ type: 'transcript', turn });
    ws.push({
      type: 'card',
      card: {
        mode: 'recall',
        status: 'partial',
        findings: [{ ...finding, kind: 'consistency', label: 'contradiction' }],
      },
    });
    expect(screen.getByText('Contradiction')).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText(/plain wording/i));
    expect(screen.getByText(/does not match/i)).toBeInTheDocument();
  });
});
