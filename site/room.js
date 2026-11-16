(() => {
  /* Watch it work: a scripted 80-second meeting (room-data.json) played on a virtual clock.
     A card appears on its own at each marked moment; Space asks what matters at any point.
     The real-meeting replay and the live pipeline live on lab.html. Every string reaches the page through textContent. */
  const root = document.querySelector('[data-demo]');
  if (!root) return;
  const $ = (s) => root.querySelector(s);
  const el = (tag, cls, text, kids) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    (kids || []).forEach((k) => n.append(k));
    return n;
  };
  const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fmt = (ms) => { const s = Math.max(0, Math.floor(ms / 1000)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
  const TEAM = 'team_routed';
  const STATUS = { answered: 'Answered', partial: 'Partly answered', abstained: 'Not discussed yet' };
  const LABELS = { update: 'Changed', contradiction: 'Contradiction' };
  /* short names for the moments on the rail, keyed by the scripted press */
  const TITLES = {
    'ship@23000': 'The ship date moves',
    'budget@33000': 'A budget is approved',
    'qa@38000': 'Who owns the QA plan?',
    'budget@59000': 'The budget is overturned',
    'press@80000': 'Who takes the press list?'
  };
  const FIRST_MOMENT_MS = 10000; /* the scripted press at 0:04 is too early to be interesting */

  const cast = window.CLAvatar ? window.CLAvatar.cast() : null;
  const face = (speaker, cls) => {
    const w = el('span', 'face' + (cls ? ' ' + cls : ''));
    if (cast) w.append(window.CLAvatar.make(cast.key(speaker)));
    return w;
  };

  const ui = {
    stage: $('[data-stage]'), tiles: $('[data-tiles]'), cc: $('[data-cc]'), cards: $('[data-cards]'), bigplay: $('[data-bigplay]'),
    play: $('[data-play]'), seek: $('[data-seek]'), marks: $('[data-marks]'), time: $('[data-time]'), speed: $('[data-speed]'),
    press: $('[data-press]'), rail: $('[data-rail]'), turns: $('[data-turns]'), credit: $('[data-credit]')
  };
  const st = {
    data: null, turns: [], moments: [], duration: 0,
    base: 0, wall: 0, playing: false, speed: 1, started: false,
    showing: undefined, manual: false, cited: new Set(), rows: new Map(), shownTurns: 0
  };

  /* ---- clock ---- */
  const pos = () => Math.min(st.duration, st.base + (st.playing ? (performance.now() - st.wall) * st.speed : 0));
  function play() {
    if (pos() >= st.duration) { st.base = 0; st.showing = undefined; }
    st.wall = performance.now();
    st.playing = st.started = true;
    tick();
  }
  function pause() { st.base = pos(); st.playing = false; tick(); }
  function seek(ms) { st.base = Math.max(0, Math.min(st.duration, ms)); st.wall = performance.now(); st.manual = false; st.showing = undefined; tick(); }
  function setSpeed(n) {
    st.base = pos(); st.wall = performance.now(); st.speed = n;
    ui.speed.setAttribute('aria-pressed', String(n === 2));
    ui.speed.textContent = n === 2 ? '2× speed' : '1× speed';
  }

  /* ---- the call ---- */
  function buildTiles() {
    const names = [...new Set(st.turns.map((t) => t.speaker))].slice(0, 8);
    ui.tiles.replaceChildren(...names.map((n) => {
      const bars = el('span', 'tile__bars', null, [el('span'), el('span'), el('span')]);
      const li = el('li', 'tile', null, [face(n, 'tile__face'), el('span', 'tile__name', null, [el('span', null, n), bars])]);
      li.dataset.who = n;
      return li;
    }));
    ui.tiles.style.setProperty('--cols', String(names.length <= 2 ? Math.max(names.length, 1) : names.length <= 4 ? 2 : 3));
  }
  function renderCall(now) {
    const talking = st.started ? st.turns.find((t) => t.start_ms <= now && now <= t.end_ms + 300) : null;
    [...ui.tiles.children].forEach((n) => n.classList.toggle('is-talking', !!talking && st.playing && talking.speaker === n.dataset.who));
    const key = talking ? talking.id : '';
    if (ui.cc.dataset.k !== key) {
      ui.cc.dataset.k = key;
      ui.cc.replaceChildren(...(talking ? [el('b', null, talking.speaker + ' '), document.createTextNode(talking.text)] : []));
    }
  }

  /* ---- transcript ---- */
  function turnRow(t) {
    const li = el('li', st.cited.has(t.id) ? 'is-cited' : '', null, [
      face(t.speaker),
      el('div', 'chat__body', null, [
        el('div', 'chat__meta', null, [el('span', 'chat__who', t.speaker), el('span', 'measured chat__at', fmt(t.start_ms)), el('span', 'chat__tag', 'cited')]),
        el('p', 'chat__text', t.text)
      ])
    ]);
    st.rows.set(t.id, li);
    return li;
  }
  function renderTurns(now) {
    const n = st.turns.filter((t) => t.start_ms <= now).length;
    if (n === st.shownTurns) return;
    if (n < st.shownTurns) { ui.turns.replaceChildren(); st.rows.clear(); st.shownTurns = 0; }
    ui.turns.append(...st.turns.slice(st.shownTurns, n).map(turnRow));
    st.shownTurns = n;
    ui.turns.scrollTop = ui.turns.scrollHeight;
  }
  function cite(ids) {
    st.cited = new Set(ids);
    st.rows.forEach((li, id) => li.classList.toggle('is-cited', st.cited.has(id)));
  }

  /* ---- cards ---- */
  function cardEl(res, trigger) {
    const c = res.card;
    const chip = el('span', 'chip ' + (c.status === 'answered' ? 'chip--ok' : c.status === 'partial' ? 'chip--warn' : 'chip--mute'), STATUS[c.status] || c.status);
    const kids = [
      el('p', 'rcard__trigger rcard__trigger--' + trigger.kind, trigger.text),
      el('div', 'rcard__top', null, [el('span', 'rcard__brand', null, [el('span', 'rcard__dot'), el('span', 'rcard__name', 'ContextLens')]), chip])
    ];
    if (c.findings.length) {
      const quoted = new Set(); /* each turn is quoted once per card */
      kids.push(el('ul', null, null, c.findings.map((f) => {
        const lab = f.label && f.label !== 'consistent' ? [el('span', 'chip ' + (f.label === 'contradiction' ? 'chip--bad' : 'chip--warn'), LABELS[f.label] || f.label)] : [];
        const fresh = f.evidence.filter((e) => !quoted.has(e.turn_id)).slice(0, 2);
        fresh.forEach((e) => quoted.add(e.turn_id));
        const quotes = fresh.map((e) => el('div', 'rcard__q', null, [
          face(e.speaker, 'face--sm'),
          el('span', null, null, [el('b', null, e.speaker + ' '), document.createTextNode('“' + e.quote + '”')])
        ]));
        return el('li', null, null, [el('div', 'rcard__stmt', null, [...lab, el('span', null, f.statement)]), ...quotes]);
      })));
    }
    if (c.status === 'abstained') kids.push(el('p', 'rcard__why', 'Nobody has said this yet, so the card says so instead of guessing.'));
    if (c.suggestion) kids.push(el('p', 'rcard__suggest', null, [el('b', null, 'Next '), document.createTextNode(c.suggestion)]));
    return el('article', 'rcard', null, kids);
  }
  function idleCard() {
    return el('article', 'rcard rcard--idle', null, [
      el('div', 'rcard__top', null, [el('span', 'rcard__brand', null, [el('span', 'rcard__dot'), el('span', 'rcard__name', 'ContextLens')]), el('span', 'chip chip--mute', st.started ? 'Listening' : 'Ready')]),
      el('p', 'rcard__why', st.started
        ? 'A card appears here when something changes or gets decided. Press Space to ask what matters right now.'
        : 'Press Play. Cards appear here on their own as the meeting goes.')
    ]);
  }
  function show(res, trigger) {
    ui.cards.replaceChildren(res ? cardEl(res, trigger) : idleCard());
    cite(res ? res.card.findings.flatMap((f) => f.evidence.map((e) => e.turn_id)) : []);
  }
  const teamResult = (press) => press.results.find((r) => r.assistant === TEAM) || press.results[0];
  function momentTrigger(m) {
    return m.auto
      ? { kind: 'auto', text: 'Noticed on its own at ' + fmt(m.at) }
      : { kind: 'press', text: 'Press at ' + fmt(m.at) + ': “' + m.press.label + '”' };
  }

  /* the card follows the latest moment already reached, unless Space asked for something since */
  function renderCard(now) {
    let idx = -1;
    st.moments.forEach((m, i) => { if (m.at <= now) idx = i; });
    if (idx === st.showing) return;
    st.showing = idx;
    if (idx < 0) show(null); else show(teamResult(st.moments[idx].press), momentTrigger(st.moments[idx]));
    renderRail(idx);
  }
  function pressNow() {
    if (!st.data) return;
    const now = pos();
    const best = st.data.presses.filter((p) => p.after_ms <= now).sort((a, b) => b.after_ms - a.after_ms)[0];
    const res = best ? teamResult(best) : st.data.abstain[TEAM];
    st.manual = true;
    st.started = true;
    show(res, { kind: 'press', text: 'You pressed at ' + fmt(now) + ': what matters now?' });
    tick();
  }

  /* ---- the moments rail ---- */
  function buildRail() {
    ui.rail.replaceChildren(...st.moments.map((m) => {
      const b = el('button', 'moment moment--' + (m.auto ? 'auto' : 'press'), null, [
        el('span', 'moment__time measured', fmt(m.at)), el('span', 'moment__title', m.title),
        el('span', 'moment__kind', m.auto ? 'Appears on its own' : 'What a press shows')
      ]);
      b.type = 'button';
      b.addEventListener('click', (e) => {
        if (e.detail) b.blur();
        st.started = true;
        seek(m.at);
        ui.stage.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' });
      });
      return el('li', null, null, [b]);
    }));
    ui.marks.replaceChildren(...st.moments.map((m) => {
      const k = el('span', 'track__mark track__mark--' + (m.auto ? 'warn' : 'cool'));
      k.style.left = (m.at / st.duration) * 100 + '%';
      return k;
    }));
  }
  function renderRail(idx) {
    [...ui.rail.children].forEach((li, i) => {
      const b = li.firstChild;
      b.classList.toggle('is-past', i < idx);
      if (i === idx) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
    });
  }

  /* ---- the loop ---- */
  let lastSec = -1;
  function tick() {
    if (!st.data) return;
    let now = pos();
    if (st.playing && now >= st.duration) { st.playing = false; st.base = st.duration; now = st.duration; }
    const ended = now >= st.duration && !st.playing;
    ui.play.textContent = st.playing ? 'Pause' : ended ? 'Replay' : 'Play';
    ui.play.setAttribute('aria-pressed', String(st.playing));
    ui.bigplay.hidden = st.started;
    if (document.activeElement !== ui.seek) ui.seek.value = String(Math.round(now / 1000));
    if (Math.floor(now / 1000) !== lastSec || !st.playing) {
      lastSec = Math.floor(now / 1000);
      ui.time.textContent = fmt(now) + ' / ' + fmt(st.duration);
    }
    renderCall(now);
    renderTurns(now);
    if (!st.manual) renderCard(now);
    else {
      /* a Space card stays until the meeting reaches the next moment */
      const next = st.moments.find((m) => m.at > (st.manualAt == null ? now : st.manualAt));
      if (st.manualAt == null) st.manualAt = now;
      if (next && now >= next.at) { st.manual = false; st.manualAt = null; renderCard(now); }
    }
  }
  setInterval(() => { if (st.playing) tick(); }, 200);

  /* ---- controls ---- */
  ui.bigplay.addEventListener('click', () => { play(); ui.play.focus(); });
  ui.play.addEventListener('click', (e) => { if (e.detail) ui.play.blur(); st.playing ? pause() : play(); });
  ui.speed.addEventListener('click', (e) => { if (e.detail) ui.speed.blur(); setSpeed(st.speed === 2 ? 1 : 2); });
  ui.press.addEventListener('click', (e) => { if (e.detail) ui.press.blur(); st.manualAt = null; pressNow(); });
  ui.seek.addEventListener('input', () => { st.started = true; st.manualAt = null; seek(Number(ui.seek.value) * 1000); });
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest && e.target.closest('input, textarea, select, button, summary, a')) return;
    e.preventDefault();
    st.manualAt = null;
    pressNow();
  });

  /* ---- load ---- */
  fetch('room-data.json').then((r) => r.json()).then((d) => {
    st.data = d;
    st.turns = d.turns;
    st.duration = d.duration_ms;
    const autoAt = new Set(d.flags.map((f) => f.at_ms));
    st.moments = d.presses.filter((p) => p.after_ms >= FIRST_MOMENT_MS).sort((a, b) => a.after_ms - b.after_ms).map((p) => ({
      at: p.after_ms, press: p, auto: autoAt.has(p.after_ms), title: TITLES[p.key + '@' + p.after_ms] || p.label
    }));
    ui.seek.max = String(Math.floor(st.duration / 1000));
    ui.bigplay.lastChild.textContent = 'Play the demo · ' + fmt(st.duration);
    ui.credit.textContent = 'An invented meeting with prepared cards, so the demo runs in your browser with no model.';
    buildTiles();
    buildRail();
    show(null);
    tick();
  }).catch(() => { ui.cards.replaceChildren(el('article', 'rcard', null, [el('p', 'rcard__why', 'The demo meeting could not be loaded.')])); });
})();
