import { parseServerMessage, audioMessage, pressMessage } from './protocol.js';

// One WebSocket connection to /ws/{conversation_id}. Same state rules as clients/web/src/session.ts:
// audio leaves only while the client believes it is recording, and a stale `recording` status after
// pause, delete or close is ignored.
export class Session {
  constructor(url, conversationId, { Socket = globalThis.WebSocket, turns = [], cards = [] } = {}) {
    this.conversationId = conversationId;
    this.listeners = new Set();
    this.resumePending = false;
    this.state = { connection: 'connecting', error: null, turns, cards, card: cards.length ? cards[cards.length - 1].card : null, waiting: false, fault: false, memory: [] };
    this.ws = new Socket(url);
    this.ws.onmessage = (e) => this.receive(e.data);
    this.ws.onclose = () => {
      if (this.state.connection === 'deleted') return;
      this.resumePending = false;
      this.set({ connection: 'closed', waiting: false });
    };
    this.ws.onerror = () => this.set({ error: 'Could not reach the server.' });
  }

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  set(patch) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((fn) => fn(this.state));
  }

  receive(raw) {
    const msg = parseServerMessage(raw);
    if (!msg) return;
    if (msg.type !== 'status' && this.state.connection === 'deleted') return;
    if (msg.type === 'transcript') {
      this.set({ turns: [...this.state.turns, msg.turn] });
    } else if (msg.type === 'turn_update') {
      if (!this.state.turns.some((t) => t.id === msg.turn.id)) return;
      this.set({ turns: this.state.turns.map((t) => (t.id === msg.turn.id ? msg.turn : t)) });
    } else if (msg.type === 'card') {
      const entry = { card: msg.card, at: Date.now(), question: this.lastQuestion ?? null };
      this.set({ card: msg.card, cards: [...this.state.cards, entry], waiting: false });
    } else if (msg.type === 'memory') {
      this.set({ memory: msg.items });
    } else if (msg.state === 'error') {
      const fault = this.state.fault || this.state.connection === 'recording';
      this.set({ error: msg.detail ?? 'The server reported an error.', waiting: false, fault });
    } else if (msg.state === 'deleted') {
      this.set({ connection: 'deleted', turns: [], cards: [], card: null, memory: [], waiting: false, error: null, fault: false });
    } else {
      this.receiveStatus(msg.state);
    }
  }

  receiveStatus(state) {
    const c = this.state.connection;
    if (c === 'deleting' || c === 'deleted' || c === 'closed') return;
    if (state === 'recording') {
      if (c === 'paused' && !this.resumePending) return;
      this.resumePending = false;
      this.set({ connection: 'recording', error: null, fault: false });
    } else if (c !== 'paused') {
      this.set({ connection: 'paused', error: null });
    }
  }

  send(msg) {
    if (this.ws.readyState === 1) this.ws.send(JSON.stringify(msg));
  }

  sendAudio(data, sampleRate = 16000) {
    if (this.state.connection !== 'recording' || this.state.fault) return;
    this.send(audioMessage(data, sampleRate));
  }

  // Returns false when there is nothing to press on yet.
  press(detail, question) {
    const last = this.state.turns[this.state.turns.length - 1];
    if (!last || this.ws.readyState !== 1) return false;
    this.lastQuestion = question || null;
    this.set({ waiting: true, error: this.state.fault ? this.state.error : null });
    this.send(pressMessage(this.conversationId, last.seq, detail, question));
    return true;
  }

  pause() {
    if (this.state.connection !== 'recording') return;
    this.resumePending = false;
    this.set({ connection: 'paused' });
    this.send({ type: 'pause' });
  }

  resume() {
    if (this.state.connection !== 'paused' || this.resumePending) return;
    this.resumePending = true;
    this.send({ type: 'resume' });
  }

  delete() {
    const c = this.state.connection;
    if (c === 'deleting' || c === 'deleted' || c === 'closed') return;
    this.resumePending = false;
    this.set({ connection: 'deleting' });
    this.send({ type: 'delete' });
  }

  close() {
    this.ws.onclose = null;
    this.ws.close();
  }
}
