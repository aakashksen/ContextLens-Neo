import net from 'node:net';

export const MODELS = ['qwen3.5:4b-q4_K_M', 'nomic-embed-text'];

// Each check resolves to { id, label, state: 'ready' | 'missing' | 'warming', detail, fix } and never throws.
// Checks only look: a TCP connect, the Ollama model list and the API's /ready. No model is run.
export function tcpOpen(host, port, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const s = net.connect({ host, port });
    const done = (ok) => {
      s.destroy();
      resolve(ok);
    };
    s.setTimeout(timeoutMs, () => done(false));
    s.once('connect', () => done(true));
    s.once('error', () => done(false));
  });
}

const httpBase = (wsUrl) => wsUrl.replace(/^ws/, 'http').replace(/\/+$/, '');

export async function checkAll({
  serverUrl,
  ollamaUrl = 'http://127.0.0.1:11434',
  db = ['127.0.0.1', 5432],
  speech = ['127.0.0.1', 50051],
  fetchFn = fetch,
  tcp = tcpOpen,
  models = MODELS,
} = {}) {
  const ollama = (async () => {
    try {
      const r = await fetchFn(`${ollamaUrl}/api/tags`, { signal: AbortSignal.timeout(2000) });
      const have = ((await r.json()).models ?? []).map((m) => m.name);
      const missing = models.filter((m) => !have.some((h) => h === m || h === `${m}:latest`));
      if (missing.length) {
        return { id: 'ollama', label: 'Ollama models', state: 'missing', detail: `missing ${missing.join(', ')}`, fix: missing.map((m) => `ollama pull ${m}`).join(' ; ') };
      }
      return { id: 'ollama', label: 'Ollama models', state: 'ready', detail: 'both models present', fix: null };
    } catch {
      return { id: 'ollama', label: 'Ollama models', state: 'missing', detail: 'Ollama is not reachable', fix: 'ollama serve' };
    }
  })();
  const port = async (id, label, [host, p], fix) =>
    (await tcp(host, p))
      ? { id, label, state: 'ready', detail: `${host}:${p}`, fix: null }
      : { id, label, state: 'missing', detail: `nothing on ${host}:${p}`, fix };
  const api = (async () => {
    try {
      const r = await fetchFn(`${httpBase(serverUrl)}/ready`, { signal: AbortSignal.timeout(2000) });
      if (r.status === 200) return { id: 'api', label: 'Backend API', state: 'ready', detail: 'ready', fix: null };
      return { id: 'api', label: 'Backend API', state: 'warming', detail: 'loading models', fix: null };
    } catch {
      return { id: 'api', label: 'Backend API', state: 'missing', detail: 'not reachable', fix: 'Press "Start backend" (runs: python -m contextlens.api.production)' };
    }
  })();
  return Promise.all([
    ollama,
    port('speech', 'Speech service', speech, 'docker compose up -d speech'),
    port('db', 'PostgreSQL', db, 'docker compose up -d db'),
    api,
  ]);
}
