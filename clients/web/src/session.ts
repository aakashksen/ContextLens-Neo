import type { Card, ClientMessage, Detail, Turn } from './protocol';
import { parseServerMessage } from './protocol';

export type Connection = 'connecting' | 'recording' | 'paused' | 'deleting' | 'deleted' | 'closed';

export interface SessionState {
  connection: Connection;
  error: string | null;
  turns: Turn[];
  card: Card | null;
  waiting: boolean;
}

export type SocketCtor = new (url: string) => WebSocket;

export class Session {
  private ws: WebSocket;
  private listeners = new Set<() => void>();
  private state: SessionState = { connection: 'connecting', error: null, turns: [], card: null, waiting: false };

  constructor(
    url: string,
    private conversationId: string,
    Socket: SocketCtor = WebSocket,
  ) {
    this.ws = new Socket(url);
    this.ws.onmessage = (e) => this.receive(e.data);
    this.ws.onclose = () => {
      if (this.state.connection !== 'deleted') this.set({ connection: 'closed', waiting: false });
    };
    this.ws.onerror = () => this.set({ error: 'Could not reach the server.' });
  }

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  };

  getSnapshot = () => this.state;

  private set(patch: Partial<SessionState>) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((fn) => fn());
  }

  private receive(raw: unknown) {
    const msg = parseServerMessage(raw);
    if (!msg) return;
    if (msg.type === 'transcript') {
      this.set({ turns: [...this.state.turns, msg.turn] });
    } else if (msg.type === 'card') {
      this.set({ card: msg.card, waiting: false });
    } else if (msg.state === 'error') {
      this.set({ error: msg.detail ?? 'The server reported an error.', waiting: false });
    } else if (msg.state === 'deleted') {
      this.set({ connection: 'deleted', turns: [], card: null, waiting: false, error: null });
    } else if (this.state.connection === 'deleting' || this.state.connection === 'closed') {
      // A late status must not restart capture after delete or disconnect.
    } else {
      this.set({ connection: msg.state, error: null });
    }
  }

  private send(msg: ClientMessage) {
    if (this.ws.readyState === 1) this.ws.send(JSON.stringify(msg));
  }

  // Audio leaves only while the client itself believes it is recording.
  sendAudio(data: string, sampleRate = 16000) {
    if (this.state.connection !== 'recording') return;
    this.send({ type: 'audio', data, sample_rate: sampleRate });
  }

  press(detail: Detail) {
    const last = this.state.turns[this.state.turns.length - 1];
    if (!last) return;
    this.set({ waiting: true, error: null });
    this.send({
      type: 'press',
      request: { conversation_id: this.conversationId, as_of_seq: last.seq, detail },
    });
  }

  pause() {
    if (this.state.connection !== 'recording') return;
    this.set({ connection: 'paused' });
    this.send({ type: 'pause' });
  }

  resume() {
    this.send({ type: 'resume' });
  }

  delete() {
    this.set({ connection: 'deleting' });
    this.send({ type: 'delete' });
  }

  close() {
    this.ws.onclose = null;
    this.ws.close();
  }
}
