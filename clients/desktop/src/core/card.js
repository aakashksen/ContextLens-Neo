// Card rendering shared by the overlay and the main window. DOM is built with textContent only.
export const STATUS_TEXT = {
  answered: 'Answered',
  partial: 'Partial answer',
  abstained: 'Not enough information',
};
export const LABEL_TEXT = {
  consistent: 'Matches what was said earlier',
  update: 'Changes what was said earlier',
  contradiction: 'Conflicts with what was said earlier',
  insufficient: 'Not enough to tell',
};

function el(doc, tag, cls, text) {
  const e = doc.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

export function formatMs(ms) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

// detail: brief = first finding, no quotes; standard = findings with quote and speaker; detailed = also suggestion, gaps, numbers.
export function renderCard(doc, card, { detail = 'standard' } = {}) {
  const root = el(doc, 'article', `card card-${card.status}`);
  root.setAttribute('aria-label', 'Answer');
  root.append(el(doc, 'h2', 'status', STATUS_TEXT[card.status] ?? card.status));
  const findings = detail === 'brief' ? card.findings.slice(0, 1) : card.findings;
  const list = el(doc, 'ul', 'findings');
  for (const f of findings) {
    const li = el(doc, 'li', 'finding');
    if (f.label) li.append(el(doc, 'span', `badge badge-${f.label}`, LABEL_TEXT[f.label] ?? f.label));
    li.append(el(doc, 'p', 'statement', f.statement));
    if (detail !== 'brief') {
      for (const ev of f.evidence ?? []) {
        const q = el(doc, 'blockquote', 'quote');
        q.append(el(doc, 'span', 'quote-text', `“${ev.quote}”`));
        q.append(el(doc, 'cite', 'quote-by', `${ev.speaker} at ${formatMs(ev.start_ms)}`));
        li.append(q);
      }
    }
    list.append(li);
  }
  if (findings.length) root.append(list);
  if (card.suggestion && detail !== 'brief') root.append(el(doc, 'p', 'suggestion', card.suggestion));
  if (card.missing?.length && detail !== 'brief') root.append(el(doc, 'p', 'missing', `Missing: ${card.missing.join('; ')}`));
  if (detail === 'detailed') {
    const bits = [];
    if (card.latency_ms) bits.push(`${(card.latency_ms / 1000).toFixed(1)} s`);
    if (card.tokens) bits.push(`${card.tokens} tokens`);
    if (bits.length) root.append(el(doc, 'p', 'numbers', bits.join(' · ')));
  }
  return root;
}

// Normalises the optional card.trace the backend may send. Returns null when there is nothing to show.
export function traceModel(card) {
  const t = card?.trace;
  const stages = [];
  if (Array.isArray(t?.stages)) for (const s of t.stages) stages.push({ name: String(s.name), ms: Number(s.ms) });
  else if (t?.stage_ms && typeof t.stage_ms === 'object') for (const [name, ms] of Object.entries(t.stage_ms)) stages.push({ name, ms: Number(ms) });
  const specialists = [].concat(t?.specialist ?? t?.specialists ?? []).map(String);
  const verdict = t?.checker?.verdict ?? t?.checker_verdict ?? null;
  const drops = (t?.guard_drops ?? []).map((d) => (typeof d === 'string' ? d : (d.reason ?? JSON.stringify(d))));
  if (!stages.length && !specialists.length && !verdict && !drops.length) return null;
  return { stages, specialists, verdict, drops, total: card.latency_ms || null };
}

export function renderTrace(doc, card) {
  const root = el(doc, 'section', 'trace');
  root.append(el(doc, 'h3', null, 'How it got here'));
  const m = traceModel(card);
  if (!m) {
    root.append(el(doc, 'p', 'muted', 'The backend sent no trace for this card.'));
    if (card.latency_ms) root.append(el(doc, 'p', 'muted', `Total ${(card.latency_ms / 1000).toFixed(1)} s, ${card.tokens ?? 0} tokens.`));
    return root;
  }
  const dl = el(doc, 'dl');
  const row = (k, v) => dl.append(el(doc, 'dt', null, k), el(doc, 'dd', null, v));
  for (const s of m.stages) row(s.name, `${s.ms} ms`);
  if (m.total) row('total', `${m.total} ms`);
  if (m.specialists.length) row('specialist', m.specialists.join(', '));
  if (m.verdict) row('checker', String(m.verdict));
  row('guard drops', m.drops.length ? m.drops.join('; ') : 'none');
  root.append(dl);
  return root;
}
