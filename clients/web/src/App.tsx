import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { startMic as realStartMic } from './audio';
import type { StopMic } from './audio';
import CardView from './components/CardView';
import DeleteConfirm from './components/DeleteConfirm';
import RecordingIndicator from './components/RecordingIndicator';
import type { IndicatorState } from './components/RecordingIndicator';
import Settings from './components/Settings';
import type { SettingsValue } from './components/Settings';
import { Session } from './session';
import type { SocketCtor } from './session';

interface Props {
  socketCtor?: SocketCtor;
  startMic?: (onChunk: (b64: string, rate: number) => void, onEnded: () => void) => Promise<StopMic>;
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
  const [capture, setCapture] = useState<'off' | 'starting' | 'on' | 'blocked' | 'stopped'>('off');
  const deleteButton = useRef<HTMLButtonElement>(null);
  const root = useRef<HTMLElement>(null);

  const recording = state.connection === 'recording';
  const capturing = recording && !state.fault;
  useEffect(() => {
    if (!capturing) return;
    let stop: StopMic | null = null;
    let cancelled = false;
    setCapture('starting');
    startMic(
      (b64, rate) => !cancelled && session.sendAudio(b64, rate),
      () => !cancelled && setCapture('stopped'),
    ).then(
      (s) => {
        if (cancelled) return s();
        stop = s;
        setCapture('on');
      },
      () => !cancelled && setCapture('blocked'),
    );
    return () => {
      cancelled = true;
      stop?.();
      setCapture('off');
    };
  }, [capturing, session, startMic]);

  // After the delete dialog closes, put focus back on its opener, or on the settings if that is disabled now.
  const wasConfirming = useRef(false);
  useEffect(() => {
    if (wasConfirming.current && !confirming) {
      const opener = deleteButton.current;
      (opener && !opener.disabled ? opener : root.current?.querySelector<HTMLElement>('.settings select'))?.focus();
    }
    wasConfirming.current = confirming;
  }, [confirming]);

  const live = state.connection === 'recording' || state.connection === 'paused';
  const error = state.error ?? (capture === 'blocked' ? 'The microphone is not available.' : null);
  let indicator: IndicatorState = state.connection;
  if (state.connection === 'recording') {
    if (state.error || state.fault) indicator = 'error';
    else if (capture === 'on') indicator = 'recording';
    else if (capture === 'blocked') indicator = 'blocked';
    else if (capture === 'stopped') indicator = 'mic-stopped';
    else indicator = 'starting';
  } else if (state.connection === 'connecting' && state.error) indicator = 'error';

  return (
    <main ref={root} className={`app size-${settings.size}`}>
      <header>
        <h1>ContextLens Neo</h1>
        <RecordingIndicator state={indicator} />
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
        <button type="button" ref={deleteButton} disabled={!live} onClick={() => setConfirming(true)}>
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
