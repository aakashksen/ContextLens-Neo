import { useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import { startMic as realStartMic } from './audio';
import type { StopMic } from './audio';
import CardView from './components/CardView';
import DeleteConfirm from './components/DeleteConfirm';
import RecordingIndicator from './components/RecordingIndicator';
import Settings from './components/Settings';
import type { SettingsValue } from './components/Settings';
import { Session } from './session';
import type { SocketCtor } from './session';

interface Props {
  socketCtor?: SocketCtor;
  startMic?: (onChunk: (b64: string, rate: number) => void) => Promise<StopMic>;
  serverUrl?: string;
  conversationId?: string;
}

function randomId(): string {
  return `web-${crypto.randomUUID()}`;
}

export default function App({ socketCtor, startMic, serverUrl, conversationId }: Props) {
  const convId = useMemo(
    () => conversationId ?? new URLSearchParams(window.location.search).get('conversation') ?? randomId(),
    [conversationId],
  );
  const [session, setSession] = useState<Session | null>(null);

  // Created in an effect so StrictMode's mount/unmount/mount closes only the first socket.
  useEffect(() => {
    const base = serverUrl ?? new URLSearchParams(window.location.search).get('server') ?? 'ws://127.0.0.1:8000';
    const s = new Session(`${base}/ws/${convId}`, convId, socketCtor);
    setSession(s);
    return () => s.close();
  }, [serverUrl, convId, socketCtor]);

  return session && <Live session={session} startMic={startMic ?? realStartMic} />;
}

function Live({ session, startMic }: { session: Session; startMic: NonNullable<Props['startMic']> }) {
  const state = useSyncExternalStore(session.subscribe, session.getSnapshot);
  const [settings, setSettings] = useState<SettingsValue>({ detail: 'standard', plain: false, size: 'medium' });
  const [confirming, setConfirming] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);

  const recording = state.connection === 'recording';
  useEffect(() => {
    if (!recording) return;
    let stop: StopMic | null = null;
    let cancelled = false;
    startMic((b64, rate) => session.sendAudio(b64, rate)).then(
      (s) => (cancelled ? s() : (stop = s)),
      () => setMicError('The microphone is not available.'),
    );
    return () => {
      cancelled = true;
      stop?.();
    };
  }, [recording, session, startMic]);

  const live = state.connection === 'recording' || state.connection === 'paused';
  const error = state.error ?? micError;

  return (
    <main className={`app size-${settings.size}`}>
      <header>
        <h1>ContextLens</h1>
        <RecordingIndicator connection={state.connection} />
      </header>

      <div className="controls">
        <button
          type="button"
          className="primary"
          disabled={!live || state.turns.length === 0 || state.waiting}
          onClick={() => session.press(settings.detail)}
        >
          {state.waiting ? 'Working...' : 'What was said?'}
        </button>
        {state.connection === 'paused' ? (
          <button type="button" onClick={() => session.resume()}>
            Resume
          </button>
        ) : (
          <button type="button" disabled={!recording} onClick={() => session.pause()}>
            Pause
          </button>
        )}
        <button type="button" disabled={!live} onClick={() => setConfirming(true)}>
          Delete conversation
        </button>
      </div>

      {confirming && (
        <DeleteConfirm
          onCancel={() => setConfirming(false)}
          onConfirm={() => {
            setConfirming(false);
            session.delete();
          }}
        />
      )}

      {error && <p role="alert" className="error">{error}</p>}
      {state.connection === 'deleted' && <p>The conversation was deleted. Reload the page to start a new one.</p>}
      {state.connection === 'closed' && <p>Disconnected from the server. Reload the page to reconnect.</p>}

      {state.card && <CardView card={state.card} plain={settings.plain} />}

      <section aria-label="Transcript" className="transcript">
        <h2>Transcript</h2>
        <ol>
          {state.turns.map((t) => (
            <li key={t.id}>
              <strong>{t.speaker}</strong> {t.text}
            </li>
          ))}
        </ol>
      </section>

      <Settings value={settings} onChange={setSettings} />
    </main>
  );
}
