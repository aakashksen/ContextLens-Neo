export const DETAILS = ['brief', 'standard', 'detailed'];
export const ASSISTANTS = ['team', 'single', 'team_shipping_changed', 'team_routed', 'single_plus', 'team_rag', 'team_rag_nomem'];
export const CORNERS = ['top-right', 'top-left', 'bottom-right', 'bottom-left'];

export const DEFAULTS = {
  serverUrl: 'ws://127.0.0.1:8000',
  conversationId: '',
  hotkey: 'Ctrl+Shift+Space',
  askHotkey: 'Ctrl+Shift+Q',
  mic: true,
  loopback: false,
  assistant: 'team_rag', // applied when the app starts the backend (CONTEXTLENS_ASSISTANT)
  detail: 'standard',
  autoHideSeconds: 12,
  corner: 'top-right',
  repoRoot: '',
  pythonPath: '',
};

const MODIFIERS = ['Ctrl', 'Control', 'CommandOrControl', 'CmdOrCtrl', 'Alt', 'Shift', 'Super', 'Meta'];
const KEY = /^(\w|F\d{1,2}|Space|Enter|Tab|Up|Down|Left|Right|[-=,./;'`\[\]\\])$/;

// Electron accelerator: one or more modifiers plus exactly one key. Global shortcuts need a modifier.
export function validAccelerator(s) {
  if (typeof s !== 'string') return false;
  const parts = s.split('+').map((p) => p.trim());
  if (parts.some((p) => !p)) return false;
  const key = parts[parts.length - 1];
  const mods = parts.slice(0, -1);
  return mods.length > 0 && mods.every((m) => MODIFIERS.includes(m)) && !MODIFIERS.includes(key) && KEY.test(key);
}

export function validWsUrl(s) {
  try {
    return ['ws:', 'wss:'].includes(new URL(s).protocol);
  } catch {
    return false;
  }
}

// Keeps good values, ignores bad ones (keeping the previous value), and reports which keys it rejected.
export function normalizeSettings(input, base = DEFAULTS) {
  const out = { ...DEFAULTS, ...base };
  const rejected = [];
  const given = input ?? {};
  const take = (key, ok) => {
    if (!(key in given)) return;
    if (ok(given[key])) out[key] = given[key];
    else rejected.push(key);
  };
  const str = (v) => typeof v === 'string';
  take('serverUrl', (v) => str(v) && validWsUrl(v.replace(/\/+$/, '')));
  out.serverUrl = out.serverUrl.replace(/\/+$/, '');
  take('conversationId', (v) => str(v) && /^[\w-]{0,64}$/.test(v));
  take('hotkey', validAccelerator);
  take('askHotkey', validAccelerator);
  take('mic', (v) => typeof v === 'boolean');
  take('loopback', (v) => typeof v === 'boolean');
  take('assistant', (v) => ASSISTANTS.includes(v));
  take('detail', (v) => DETAILS.includes(v));
  take('autoHideSeconds', (v) => Number.isInteger(v) && v >= 2 && v <= 120);
  take('corner', (v) => CORNERS.includes(v));
  take('repoRoot', str);
  take('pythonPath', str);
  if (out.hotkey.toLowerCase() === out.askHotkey.toLowerCase()) {
    rejected.push('hotkey', 'askHotkey');
    out.hotkey = base.hotkey ?? DEFAULTS.hotkey;
    out.askHotkey = base.askHotkey ?? DEFAULTS.askHotkey;
  }
  return { settings: out, rejected };
}
