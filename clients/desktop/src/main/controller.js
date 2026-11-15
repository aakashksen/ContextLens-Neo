import { randomUUID } from 'node:crypto';
import { Session } from '../core/session.js';
import { normalizeSettings } from '../core/settings.js';
import { resolveBackend } from '../core/backend.js';

// The app's brain, free of Electron so it can be tested against tools/fake_server.py.
// It owns the connection, the settings and the backend process; windows only render its state and send commands.
export class Controller {
  constructor({ settings, saveSettings = () => {}, emit = () => {}, backend, health, devRoot = '', Socket, retryMs = 3000, setTimer = setTimeout, clearTimer = clearTimeout }) {
    this.settings = settings;
    if (!this.settings.conversationId) this.settings.conversationId = randomUUID();
    this.saveSettings = saveSettings;
    this.emit = emit;
    this.backend = backend;
    this.healthCheck = health;
    this.devRoot = devRoot;
    this.Socket = Socket;
    this.retryMs = retryMs;
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
    this.session = null;
    this.unsub = null;
    this.timer = null;
    this.wantConnected = false;
    this.userPaused = false;
    this.health = [];
    this.backendError = null;
    this.extra = { captureError: null, hotkeyErrors: [] }; // set by the Electron shell
    this.backend.onChange = () => this.publish();
  }

  snapshot() {
    const s = this.session?.state ?? { connection: 'closed', error: null, turns: this.keep.turns, cards: this.keep.cards, card: null, waiting: false, fault: false, memory: [] };
    return {
      connection: s.connection,
      error: s.error,
      turns: s.turns,
      cards: s.cards,
      waiting: s.waiting,
      memory: s.memory,
      userPaused: this.userPaused,
      capture: { active: s.connection === 'recording' && !s.fault && !this.userPaused, mic: this.settings.mic, loopback: this.settings.loopback },
      settings: this.settings,
      ...this.extra,
      health: this.health,
      backend: { running: this.backend.running, error: this.backendError, log: this.backend.log.slice(-40) },
    };
  }

  keep = { turns: [], cards: [] };

  publish() {
    this.emit('state', this.snapshot());
  }

  connect() {
    this.wantConnected = true;
    this.openSession();
  }

  openSession() {
    if (this.session) return;
    this.clearTimer(this.timer);
    const url = `${this.settings.serverUrl}/ws/${this.settings.conversationId}`;
    const session = new Session(url, this.settings.conversationId, { Socket: this.Socket, turns: this.keep.turns, cards: this.keep.cards });
    this.session = session;
    let last = session.state.connection;
    this.unsub = session.subscribe((st) => {
      if (this.session !== session) return;
      this.keep = { turns: st.turns, cards: st.cards };
      if (st.connection === 'deleted' && last !== 'deleted') return this.afterForget();
      if (st.connection === 'recording' && last !== 'recording' && this.userPaused) session.pause();
      if (st.connection === 'closed' && last !== 'closed') this.scheduleRetry();
      if (st.card && st.cards.length && st.waiting === false && st.cards.length !== this.announced) {
        this.announced = st.cards.length;
        this.emit('card', { card: st.card, question: st.cards[st.cards.length - 1].question, detail: this.settings.detail });
      }
      last = st.connection;
      this.publish();
    });
    this.announced = session.state.cards.length;
    this.publish();
  }

  scheduleRetry() {
    this.dropSession();
    if (!this.wantConnected) return;
    this.timer = this.setTimer(() => this.openSession(), this.retryMs);
  }

  dropSession() {
    this.unsub?.();
    this.session?.close();
    this.session = null;
    this.unsub = null;
  }

  disconnect() {
    this.wantConnected = false;
    this.clearTimer(this.timer);
    this.dropSession();
    this.publish();
  }

  sendAudio(data, rate) {
    this.session?.sendAudio(data, rate);
  }

  // The press. `question` is the optional typed question.
  press(question) {
    const q = typeof question === 'string' && question.trim() ? question.trim() : undefined;
    if (!this.session || !this.session.press(this.settings.detail, q)) {
      this.emit('notice', { kind: 'info', text: this.session ? 'Nothing heard yet. Say something first.' : 'Not connected to the backend.' });
      return false;
    }
    this.emit('notice', { kind: 'working', text: 'Looking back over the conversation...' });
    return true;
  }

  pause() {
    this.userPaused = true;
    this.session?.pause();
    this.publish();
  }

  resume() {
    this.userPaused = false;
    this.session?.resume();
    this.publish();
  }

  // Deletes the stored transcript on the backend, clears everything held here and starts a fresh conversation.
  forget() {
    if (!this.session) return this.afterForget();
    this.session.delete();
  }

  afterForget() {
    this.dropSession();
    this.keep = { turns: [], cards: [] };
    this.userPaused = false;
    this.settings.conversationId = randomUUID();
    this.saveSettings(this.settings);
    this.emit('notice', { kind: 'info', text: 'Conversation forgotten.' });
    if (this.wantConnected) this.openSession();
    else this.publish();
  }

  setSettings(patch) {
    const { settings, rejected } = normalizeSettings(patch, this.settings);
    const urlChanged = settings.serverUrl !== this.settings.serverUrl;
    this.settings = settings;
    this.saveSettings(settings);
    if (urlChanged && this.wantConnected) {
      this.dropSession();
      this.openSession();
    }
    this.publish();
    return { rejected };
  }

  async refreshHealth() {
    this.health = await this.healthCheck({ serverUrl: this.settings.serverUrl });
    this.publish();
    return this.health;
  }

  startBackend() {
    const r = resolveBackend({ repoRoot: this.settings.repoRoot, pythonPath: this.settings.pythonPath, devRoot: this.devRoot });
    if (r.error) {
      this.backendError = r.error;
      this.publish();
      return false;
    }
    this.backendError = null;
    this.backend.start({ ...r, assistant: this.settings.assistant });
    return true;
  }

  stopBackend() {
    this.backend.stop();
  }

  shutdown() {
    this.disconnect();
    this.backend.stop();
  }
}
