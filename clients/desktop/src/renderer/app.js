import { renderCard, renderTrace } from '../core/card.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
let state = null;
let selected = -1;

function listeningLabel(s) {
  if (s.capture.active) return 'Listening';
  if (s.userPaused || s.connection === 'paused') return 'Paused';
  if (s.connection === 'recording') return 'Starting';
  if (s.connection === 'closed') return 'Not connected';
  if (s.connection === 'deleting') return 'Forgetting';
  return 'Connecting';
}

function renderHealth(s) {
  const ul = $('health');
  ul.replaceChildren();
  for (const h of s.health) {
    const item = el('li', `health health-${h.state}`);
    item.append(el('strong', null, h.label), el('span', 'state', ` ${h.state}`), el('span', 'muted', ` ${h.detail}`));
    if (h.fix) {
      const code = el('code', null, h.fix);
      const copy = el('button', 'small', 'Copy');
      copy.addEventListener('click', () => navigator.clipboard?.writeText(h.fix));
      item.append(document.createElement('br'), code, copy);
    }
    ul.append(item);
  }
  $('start').hidden = s.backend.running;
  $('stop').hidden = !s.backend.running;
  $('backend-note').textContent = s.backend.error ?? (s.backend.running ? 'Started by this app; it stops when you quit.' : '');
  $('backend-log').hidden = !s.backend.log.length;
  $('backend-log').textContent = s.backend.log.join('\n');
}

function renderTranscript(s) {
  const ol = $('transcript');
  ol.replaceChildren(
    ...s.turns.map((t) => {
      const li = el('li', 'turn');
      li.append(el('b', t.speaker === 'unknown' ? 'speaker unknown' : 'speaker', t.speaker), el('span', null, ` ${t.text}`));
      return li;
    }),
  );
  $('transcript-empty').hidden = s.turns.length > 0;
  ol.scrollTop = ol.scrollHeight;
}

function renderMemory(s) {
  const ul = $('memory');
  ul.replaceChildren(...s.memory.map((m) => el('li', null, typeof m === 'string' ? m : (m.text ?? m.statement ?? JSON.stringify(m)))));
  $('memory-empty').hidden = s.memory.length > 0;
}

function renderCards(s) {
  const ul = $('cards');
  ul.replaceChildren(
    ...s.cards.map((c, i) => {
      const li = el('li');
      const b = el('button', i === selected ? 'on' : '', `${new Date(c.at).toLocaleTimeString()}  ${c.card.status}${c.question ? `: ${c.question}` : ''}`);
      b.addEventListener('click', () => {
        selected = i;
        renderCards(state);
      });
      li.append(b);
      return li;
    }),
  );
  $('cards-empty').hidden = s.cards.length > 0;
  const d = $('card-detail');
  d.replaceChildren();
  const c = s.cards[selected];
  if (c) d.append(renderCard(document, c.card, { detail: 'detailed' }), renderTrace(document, c.card));
}

function renderSettings(s) {
  const f = $('settings');
  if (f.contains(document.activeElement)) return;
  for (const [k, v] of Object.entries(s.settings)) {
    const input = f.elements[k];
    if (!input) continue;
    if (input.type === 'checkbox') input.checked = v;
    else input.value = v;
  }
}

function render(s) {
  state = s;
  const label = listeningLabel(s);
  $('listening').textContent = label;
  $('listening').dataset.state = label.toLowerCase();
  $('toggle').textContent = s.userPaused || s.connection === 'paused' ? 'Resume' : 'Pause';
  $('press').disabled = s.waiting;
  const problems = [s.error, s.captureError, ...(s.hotkeyErrors ?? [])].filter(Boolean);
  $('problem').hidden = problems.length === 0;
  $('problem').textContent = problems.join(' ');
  renderHealth(s);
  renderTranscript(s);
  renderMemory(s);
  renderCards(s);
  renderSettings(s);
}

for (const b of document.querySelectorAll('[role=tab]')) {
  b.addEventListener('click', () => {
    for (const t of document.querySelectorAll('[role=tab]')) {
      t.setAttribute('aria-selected', String(t === b));
      $(`tab-${t.dataset.tab}`).hidden = t !== b;
    }
  });
}
$('press').addEventListener('click', () => window.cl.invoke('press'));
$('toggle').addEventListener('click', () => window.cl.invoke(state.userPaused || state.connection === 'paused' ? 'resume' : 'pause'));
$('forget').addEventListener('click', () => {
  if (confirm("Delete this conversation's transcript and memory from this machine?")) window.cl.invoke('forget');
});
$('start').addEventListener('click', () => window.cl.invoke('start-backend'));
$('stop').addEventListener('click', () => window.cl.invoke('stop-backend'));
$('settings').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target.elements;
  const patch = {
    hotkey: f.hotkey.value.trim(),
    askHotkey: f.askHotkey.value.trim(),
    mic: f.mic.checked,
    loopback: f.loopback.checked,
    assistant: f.assistant.value,
    detail: f.detail.value,
    autoHideSeconds: Number(f.autoHideSeconds.value),
    corner: f.corner.value,
    serverUrl: f.serverUrl.value.trim(),
    repoRoot: f.repoRoot.value.trim(),
    pythonPath: f.pythonPath.value.trim(),
  };
  const { rejected } = await window.cl.invoke('set-settings', patch);
  $('settings-note').textContent = rejected.length ? `Not accepted: ${rejected.join(', ')}` : 'Saved.';
});

window.cl.on('state', render);
window.cl.invoke('get-state').then(render);
