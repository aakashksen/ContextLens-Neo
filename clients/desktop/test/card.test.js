// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { renderCard, renderTrace, traceModel } from '../src/core/card.js';
import { card, evidence, turn } from './helpers.js';

const text = (e) => e.textContent;
describe('renderCard', () => {
  it('shows status, finding, quoted turn with speaker and time', () => {
    const e = renderCard(document, card('answered', { suggestion: 'Ask Ana.' }));
    expect(e.className).toContain('card-answered');
    expect(text(e.querySelector('.status'))).toBe('Answered');
    expect(text(e.querySelector('.statement'))).toBe('Ana said Friday.');
    expect(text(e.querySelector('.quote-text'))).toContain('turn 0');
    expect(text(e.querySelector('.quote-by'))).toBe('Ana at 0:00');
    expect(text(e.querySelector('.suggestion'))).toBe('Ask Ana.');
  });
  it('shows the conflict label', () => {
    const c = card();
    c.findings[0].label = 'contradiction';
    expect(text(renderCard(document, c).querySelector('.badge-contradiction'))).toBe('Conflicts with what was said earlier');
  });
  it('partial lists what is missing; abstained has no findings', () => {
    expect(text(renderCard(document, card('partial', { missing: ['the date'] })).querySelector('.missing'))).toBe('Missing: the date');
    const a = renderCard(document, card('abstained', { missing: ['x'] }));
    expect(a.querySelector('.finding')).toBeNull();
    expect(text(a.querySelector('.status'))).toBe('Not enough information');
  });
  it('detail levels', () => {
    const c = card('answered', { suggestion: 's', latency_ms: 1500, tokens: 10 });
    c.findings.push({ ...c.findings[0], statement: 'second', evidence: [evidence(turn(1))] });
    const brief = renderCard(document, c, { detail: 'brief' });
    expect(brief.querySelectorAll('.finding')).toHaveLength(1);
    expect(brief.querySelector('.quote')).toBeNull();
    expect(brief.querySelector('.suggestion')).toBeNull();
    expect(renderCard(document, c, { detail: 'standard' }).querySelectorAll('.finding')).toHaveLength(2);
    expect(text(renderCard(document, c, { detail: 'detailed' }).querySelector('.numbers'))).toBe('1.5 s · 10 tokens');
  });
  it('never interprets card text as HTML', () => {
    const c = card();
    c.findings[0].statement = '<img src=x onerror=alert(1)>';
    const e = renderCard(document, c);
    expect(e.querySelector('img')).toBeNull();
    expect(text(e.querySelector('.statement'))).toContain('<img');
  });
});

describe('trace', () => {
  const traced = card('answered', { latency_ms: 900, trace: { stages: [{ name: 'retrieve', ms: 120 }, { name: 'draft', ms: 700 }], specialist: 'consistency', checker: { verdict: 'pass' }, guard_drops: ['no quote'] } });
  it('normalises stages, specialist, verdict and drops', () => {
    expect(traceModel(traced)).toMatchObject({ stages: [{ name: 'retrieve', ms: 120 }, { name: 'draft', ms: 700 }], specialists: ['consistency'], verdict: 'pass', drops: ['no quote'], total: 900 });
  });
  it('accepts a stage_ms map', () => expect(traceModel(card('answered', { trace: { stage_ms: { a: 1 } } })).stages).toEqual([{ name: 'a', ms: 1 }]));
  it('renders rows, and says so when there is no trace', () => {
    const t = text(renderTrace(document, traced));
    for (const w of ['retrieve', 'consistency', 'pass', 'no quote']) expect(t).toContain(w);
    expect(text(renderTrace(document, card()))).toContain('no trace');
  });
});
