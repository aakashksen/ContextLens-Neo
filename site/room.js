(() => {
  /* The meeting room. Three modes, one page:
       live    tools/meeting_room.py --live   real speech service, diarizer, stream API and local models
       replay  tools/meeting_room.py          AMI transcript with its own timings, a word-overlap stand-in for the cards
       static  no server                      a scripted invented meeting from room-data.json, played on a virtual clock
     Every string reaches the page through textContent. */
  const root = document.querySelector('[data-room]');
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
  const who = (s) => {
    const m = /^SPEAKER_(\d+)$/.exec(s || '');
    if (m) return 'Speaker ' + (Number(m[1]) + 1);
    if (!s || s === 'unknown') return 'Unattributed';
    return /^[A-Z]$/.test(s) ? 'Speaker ' + s : s;
  };
  const NAMES = { single_plus: 'Single agent', team_routed: 'Agent team', team_rag: 'Agent team' };
  const STATUS = { answered: 'Answered', partial: 'Partly answered', abstained: 'Not discussed yet' };
  const LABELS = { update: 'Changed', contradiction: 'Contradiction' };
  const PRESETS = ['What was decided?', 'Who owns what?', 'What changed?'];
  const faces = window.CLAvatar ? window.CLAvatar.cast() : { key: () => 'someone', reset() {} };
  const face = (speaker, cls) => {
    const w = el('span', 'face' + (cls ? ' ' + cls : ''));
    if (window.CLAvatar) w.append(window.CLAvatar.make(faces.key(speaker)));
    return w;
  };

  const ui = {
    mode: $('[data-mode]'), lead: $('[data-lead]'), meeting: $('[data-meeting]'), meetingLabel: $('[data-meeting-label]'), assistants: $('[data-assistants]'),
    question: $('[data-question]'), presets: $('[data-presets]'), press: $('[data-press]'), stage: $('[data-stage]'),
    video: $('[data-video]'), tiles: $('[data-tiles]'), cards: $('[data-cards]'), play: $('[data-play]'), seek: $('[data-seek]'),
    time: $('[data-time]'), status: $('[data-status]'), turns: $('[data-turns]'), memory: $('[data-memory]'), flags: $('[data-flags]'),
    traceWrap: $('[data-trace-wrap]'), traces: $('[data-traces]'), credit: $('[data-credit]'),
    capNote: $('[data-captions-note]'), memNote: $('[data-memory-note]'), flagNote: $('[data-flags-note]'),
    bigplay: $('[data-bigplay]'), nudge: $('[data-nudge]'), cc: $('[data-cc]'), marks: $('[data-marks]'), steps: $('[data-steps]')
  };

  const st = {
    mode: 'static', turns: [], memory: [], flags: [], duration: 0, meeting: null, assistant: 'team', cited: new Set(),
    session: null, since: 0, lag: null, scripted: null, busy: false, last: null, pollTimer: null,
    started: false, carded: false, seenFlags: 0, nudgeTimer: null, arms: ['single_plus', 'team_routed']
  };
  const live = () => st.mode === 'live';
  const turnById = () => new Map(st.turns.map((t) => [t.id, t]));
  const label = (id) => { const t = turnById().get(id); return t ? '#' + t.seq : id; };

  /* ---- the clock: the audio element when there is audio, else a virtual one ---- */
  const audio = new Audio();
  audio.preload = 'auto';
  const clock = { virtual: true, base: 0, wall: 0, playing: false };
  const pos = () => {
    if (!clock.virtual) return Math.round(audio.currentTime * 1000);
    return Math.min(st.duration, clock.base + (clock.playing ? performance.now() - clock.wall : 0));
  };
  const isPlaying = () => (clock.virtual ? clock.playing : !audio.paused && !audio.ended);

  function post(path, body) {
    return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then((r) => r.json().then((j) => (r.ok ? j : Promise.reject(new Error(j.error || j.detail || 'request failed')))));
  }
  const say = (text) => { ui.status.textContent = text; };

  function play() {
    st.started = true;
    if (clock.virtual) {
      if (clock.base >= st.duration) clock.base = 0;
      clock.wall = performance.now();
      clock.playing = true;
      sync();
      return;
    }
    audio.play().catch(() => say('The browser blocked playback. Press Play again.'));
  }
  function pause() {
    if (clock.virtual) {
      clock.base = pos();
      clock.playing = false;
      sync();
    } else audio.pause();
  }
  function seekTo(ms) {
    ms = Math.max(0, Math.min(st.duration, ms));
    if (clock.virtual) { clock.base = ms; clock.wall = performance.now(); } else audio.currentTime = ms / 1000;
    sync();
  }

  /* ---- rendering ---- */
  const visible = (at) => live() || at <= pos();
  let rendered = { n: 0, last: null };
  const rows = new Map();

  function turnRow(t) {
    const li = el('li', st.cited.has(t.id) ? 'is-cited' : '', null, [
      face(t.speaker),
      el('div', 'chat__body', null, [
        el('div', 'chat__meta', null, [
          el('span', 'chat__who' + (t.speaker === 'unknown' ? ' who--unknown' : ''), who(t.speaker)),
          el('span', 'measured chat__at', '#' + t.seq + ' · ' + fmt(t.start_ms)),
          el('span', 'chat__tag', 'cited')
        ]),
        el('p', 'chat__text', t.text)
      ])
    ]);
    rows.set(t.id, li);
    return li;
  }
  function renderTurns() {
    const vis = st.turns.filter((t) => visible(t.end_ms));
    const grows = vis.length >= rendered.n && (rendered.n === 0 || (vis[rendered.n - 1] && vis[rendered.n - 1].id === rendered.last));
    const near = ui.turns.scrollHeight - ui.turns.scrollTop - ui.turns.clientHeight < 48;
    if (!grows) { ui.turns.replaceChildren(); rows.clear(); rendered = { n: 0, last: null }; }
    const add = vis.slice(rendered.n);
    if (add.length) {
      ui.turns.append(...add.map(turnRow));
      rendered = { n: vis.length, last: vis[vis.length - 1].id };
      if (near) ui.turns.scrollTop = ui.turns.scrollHeight;
    }
    return vis;
  }
  function refreshRow(t) {
    const li = rows.get(t.id);
    if (!li) return;
    li.children[0].replaceWith(face(t.speaker));
    const w = li.querySelector('.chat__who');
    w.textContent = who(t.speaker);
    w.className = 'chat__who' + (t.speaker === 'unknown' ? ' who--unknown' : '');
    li.querySelector('.chat__text').textContent = t.text;
  }

  /* everyone in the meeting has a seat from the start (static and replay know the cast); live adds people as they speak */
  function renderTiles(vis) {
    const pool = live() ? vis : st.turns;
    const names = [...new Set(pool.map((t) => t.speaker))].slice(0, 8);
    const now = pos();
    const talking = isPlaying() || live() ? st.turns.find((t) => t.start_ms <= now && now <= t.end_ms + 300 && visible(t.start_ms)) : null;
    const have = [...ui.tiles.children].map((n) => n.dataset.who);
    if (have.join('|') !== names.join('|')) {
      ui.tiles.replaceChildren(...names.map((n) => {
        const bars = el('span', 'tile__bars', null, [el('span'), el('span'), el('span')]);
        const li = el('li', 'tile', null, [face(n, 'tile__face'), el('span', 'tile__name', null, [el('span', null, who(n)), bars])]);
        li.dataset.who = n;
        return li;
      }));
    }
    [...ui.tiles.children].forEach((n) => n.classList.toggle('is-talking', !!talking && talking.speaker === n.dataset.who));
    const line = talking || (live() ? vis[vis.length - 1] : null);
    const ccKey = line ? line.id + '|' + line.text : '';
    if (ui.cc.dataset.k !== ccKey) {
      ui.cc.dataset.k = ccKey;
      ui.cc.replaceChildren(...(line ? [el('b', null, who(line.speaker) + ' '), document.createTextNode(line.text)] : []));
    }
  }

  /* a tick on the timeline wherever the checker flagged a change, so there is something to aim for */
  function renderMarks() {
    const sig = st.duration + ':' + st.flags.map((f) => f.at_ms).join(',');
    if (ui.marks.dataset.sig === sig) return;
    ui.marks.dataset.sig = sig;
    ui.marks.replaceChildren(...(st.duration ? st.flags.map((f) => {
      const m = el('span', 'track__mark track__mark--' + (f.label === 'contradiction' ? 'bad' : 'warn'));
      m.style.left = Math.min(100, (f.at_ms / st.duration) * 100) + '%';
      return m;
    }) : []));
  }

  function nudge(show) {
    clearTimeout(st.nudgeTimer);
    ui.nudge.hidden = !show;
    if (show) st.nudgeTimer = setTimeout(() => { ui.nudge.hidden = true; }, 7000);
  }

  function renderSteps() {
    const step = st.carded ? 3 : st.started ? 2 : 1;
    ui.steps.querySelectorAll('[data-step]').forEach((n) => {
      const k = Number(n.dataset.step);
      n.classList.toggle('is-done', k < step);
      if (k === step) n.setAttribute('aria-current', 'step'); else n.removeAttribute('aria-current');
    });
    ui.bigplay.hidden = st.started || isPlaying();
    const card = ui.cards.childElementCount > 0;
    ui.stage.classList.toggle('has-card', card);
    const n = ui.tiles.childElementCount;
    ui.tiles.style.setProperty('--cols', String(n <= 2 ? Math.max(n, 1) : n <= 4 || card ? 2 : 3));
  }

  function citeButton(id) {
    const b = el('button', 'cite', 'turn ' + label(id));
    b.type = 'button';
    b.title = 'Show this turn in the transcript';
    b.addEventListener('click', () => cite([id]));
    return b;
  }
  function cite(ids) {
    st.cited = new Set(ids);
    rows.forEach((li, id) => li.classList.toggle('is-cited', st.cited.has(id)));
    const first = ids.map((i) => rows.get(i)).find(Boolean);
    if (first) first.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' });
  }

  let memSig = '', flagSig = '';
  function renderMemory() {
    const items = st.memory.filter((m) => visible(m.at_ms));
    const sig = items.map((m) => m.id).join('|');
    ui.memNote.textContent = items.length ? items.length + ' so far' : '';
    if (sig === memSig) return;
    memSig = sig;
    if (!items.length) { ui.memory.replaceChildren(el('li', 'is-empty', 'Nothing decided or promised yet.')); return; }
    const replaced = new Set(items.map((m) => m.supersedes).filter(Boolean));
    ui.memory.replaceChildren(...items.slice().reverse().map((m) => {
      const kind = el('span', 'chip ' + (m.kind === 'correction' ? 'chip--warn' : 'chip--cool'), m.kind);
      const text = replaced.has(m.id) ? el('s', null, m.statement) : el('span', null, m.statement);
      const b = el('button', 'cite', m.source_turn_ids.map(label).join(' '));
      b.type = 'button';
      b.title = 'Show the turns this came from';
      b.addEventListener('click', () => cite(m.source_turn_ids));
      return el('li', null, null, [el('div', 'room__item', null, [face(m.speaker, 'face--sm'), el('span', 'room__who', who(m.speaker)), kind, b]), text]);
    }));
  }
  function renderFlags() {
    const items = st.flags.filter((f) => visible(f.at_ms));
    const sig = items.map((f) => f.turn_id + f.label).join('|');
    ui.flagNote.textContent = items.length ? items.length + ' so far' : '';
    if (items.length > st.seenFlags && isPlaying() && !ui.cards.childElementCount) nudge(true);
    st.seenFlags = items.length;
    if (sig === flagSig) return;
    flagSig = sig;
    if (!items.length) { ui.flags.replaceChildren(el('li', 'is-empty', 'Nothing has contradicted or replaced an earlier statement.')); return; }
    const byId = turnById();
    ui.flags.replaceChildren(...items.slice().reverse().map((f) => {
      const chip = el('span', 'chip ' + (f.label === 'contradiction' ? 'chip--bad' : 'chip--warn'), LABELS[f.label] || f.label);
      const ids = [f.turn_id, ...(f.earlier_ids || [])];
      const b = el('button', 'cite', ids.map(label).join(' '));
      b.type = 'button';
      b.title = 'Show the turns the checker compared';
      b.addEventListener('click', () => cite(ids));
      const t = byId.get(f.turn_id);
      return el('li', null, null, [
        el('div', 'room__item', null, [...(t ? [face(t.speaker, 'face--sm'), el('span', 'room__who', who(t.speaker))] : []), chip, b]),
        el('span', null, t ? '“' + t.text + '”' : label(f.turn_id)),
        el('span', 'room__conf', 'Checker confidence ' + Number(f.confidence).toFixed(2))
      ]);
    }));
  }

  let lastSecond = -1;
  function sync() {
    const now = pos();
    const playing = isPlaying();
    ui.play.textContent = playing ? 'Pause' : 'Play';
    ui.play.setAttribute('aria-pressed', String(playing));
    if (document.activeElement !== ui.seek) ui.seek.value = String(Math.round(now / 1000));
    if (Math.floor(now / 1000) !== lastSecond || !playing) {
      lastSecond = Math.floor(now / 1000);
      ui.time.textContent = fmt(now) + ' / ' + fmt(st.duration);
    }
    const vis = renderTurns();
    renderTiles(vis);
    renderMarks();
    renderSteps();
    renderMemory();
    renderFlags();
    if (live() && st.lag != null) ui.capNote.textContent = 'about ' + (st.lag / 1000).toFixed(1) + ' s behind the audio';
    if (clock.virtual && clock.playing && now >= st.duration) { clock.playing = false; clock.base = st.duration; say('The meeting has ended.'); }
  }
  setInterval(sync, 250);

  /* ---- cards and traces ---- */
  function cardEl(res, asof) {
    const c = res.card;
    const chip = el('span', 'chip ' + (c.status === 'answered' ? 'chip--ok' : c.status === 'partial' ? 'chip--warn' : 'chip--mute'), STATUS[c.status] || c.status);
    const kids = [el('div', 'rcard__top', null, [el('span', 'rcard__brand', null, [el('span', 'rcard__dot'), el('span', 'rcard__name', NAMES[res.assistant] || res.assistant)]), chip])];
    if (c.findings.length) {
      kids.push(el('ul', null, null, c.findings.map((f) => {
        const lab = f.label && f.label !== 'consistent' ? [el('span', 'chip ' + (f.label === 'contradiction' ? 'chip--bad' : 'chip--warn'), LABELS[f.label] || f.label)] : [];
        const cites = el('div', 'rcard__cites', null, f.evidence.map((e) => citeButton(e.turn_id)));
        const e0 = f.evidence[0];
        const q = e0 ? el('div', 'rcard__q', null, [face(e0.speaker, 'face--sm'), el('span', null, null, [el('b', null, who(e0.speaker) + ' '), document.createTextNode('“' + e0.quote + '”')])]) : null;
        return el('li', null, null, [el('div', 'rcard__stmt', null, [...lab, el('span', null, f.statement)]), ...(q ? [q] : []), cites]);
      })));
    }
    if (c.status === 'abstained') kids.push(el('p', 'rcard__why', 'Nobody has said this yet: ' + (c.missing.join('; ') || 'no backed answer') + '.'));
    else if (c.missing.length) kids.push(el('p', 'rcard__why', 'Missing: ' + c.missing.join('; ') + '.'));
    if (c.suggestion) kids.push(el('p', 'rcard__suggest', null, [el('b', null, 'Suggestion '), document.createTextNode(c.suggestion)]));
    kids.push(el('span', 'rcard__asof', asof));
    return el('article', 'rcard', null, kids);
  }

  function traceEl(res) {
    const t = res.trace;
    const max = Math.max(1, ...t.stages.map((s) => s.ms || 0));
    const stages = el('ul', 'stages', null, t.stages.map((s) => {
      const li = el('li', null, null, [
        el('span', null, s.label),
        el('span', 'stages__bar', null, s.ms ? [(() => { const b = el('span'); b.style.width = Math.max(2, (s.ms / max) * 100) + '%'; return b; })()] : []),
        el('span', 'measured', s.ms == null ? 'n/a' : s.ms + ' ms')
      ]);
      li.append(el('span', 'stages__note', s.note));
      return li;
    }));
    const byAgent = new Map();
    t.findings.forEach((f) => { (byAgent.get(f.agent) || byAgent.set(f.agent, []).get(f.agent)).push(f); });
    const who_ = byAgent.size ? el('ul', null, null, [...byAgent].map(([agent, fs]) => el('li', null, null, [
      el('span', 'trace__row', null, [el('span', 'chip chip--cool', agent), el('span', 'trace__quiet', fs.length + (fs.length === 1 ? ' finding' : ' findings'))]),
      ...fs.map((f) => el('span', null, f.statement + (f.label ? ' [' + f.label + ']' : '')))
    ]))) : el('p', 'trace__quiet', 'No finding survived.');
    const specs = el('ul', null, null, t.specialists.map((s) => el('li', 'trace__row', null, [
      el('span', 'chip ' + (s.ran ? 'chip--ok' : 'chip--mute'), s.name),
      el('span', 'trace__quiet', s.ran ? (s.ms != null ? s.ms + ' ms' : 'ran') + (s.tokens ? ', ' + s.tokens + ' tokens' : '') : 'not run (code decided)')
    ])));
    const drops = t.drops.length ? el('ul', null, null, t.drops.map((d) => el('li', null, null, [
      el('span', 'trace__row', null, [el('span', 'chip chip--bad', d.reason), el('span', 'trace__quiet', d.turn_ids.map(label).join(' '))]),
      el('span', null, '“' + d.statement + '” was dropped because ' + d.why + '.')
    ]))) : el('p', 'trace__quiet', 'The evidence guard dropped nothing.');
    const kids = [
      el('h3', null, (NAMES[res.assistant] || res.assistant) + ': ' + t.total_ms + ' ms, ' + (t.tokens || 0) + ' tokens'),
      el('h4', null, 'Stages'), stages, el('h4', null, 'Specialists'), specs, el('h4', null, 'Who contributed what'), who_,
      el('h4', null, 'Dropped by the evidence guard'), drops
    ];
    if (t.checker.ran) kids.splice(5, 0, el('p', null, 'Checker verdict: ' + (t.checker.label || (t.checker.failed ? 'failed, nothing flagged' : 'consistent')) + '.'));
    if (t.abstained.yes) kids.push(el('h4', null, 'Why it abstained'), el('p', null, t.abstained.why.join('; ') || 'no verified finding'));
    else if (t.notes.length) kids.push(el('h4', null, 'Notes'), el('p', null, t.notes.join('; ')));
    return el('section', 'trace', null, kids);
  }

  function showResults(r, asof) {
    st.last = r;
    ui.cards.replaceChildren();
    ui.traces.replaceChildren();
    if (!r.results.length) {
      ui.cards.append(el('article', 'rcard', null, [el('p', 'rcard__why', r.error || 'Nothing to show.')]));
      ui.traceWrap.hidden = true;
      return;
    }
    nudge(false);
    st.carded = true;
    r.results.forEach((x) => { ui.cards.append(cardEl(x, asof)); ui.traces.append(traceEl(x)); });
    const x = el('button', 'btn rcard__x', 'Close card (Esc)');
    x.type = 'button';
    x.addEventListener('click', dismiss);
    ui.cards.append(x);
    ui.traceWrap.hidden = false;
    const first = r.results[0].card.findings[0];
    if (first) cite(r.results.flatMap((y) => y.card.findings.flatMap((f) => f.evidence.map((e) => e.turn_id))));
  }
  function dismiss() { ui.cards.replaceChildren(); cite([]); renderSteps(); }

  /* ---- press ---- */
  /* the team is whichever team arm the source offers (the scripted page has team_routed; a newer server may offer team_rag) */
  function chosen() {
    const team = st.arms.find((a) => a.startsWith('team')) || 'team_routed';
    const single = st.arms.find((a) => a.startsWith('single')) || 'single_plus';
    return st.assistant === 'both' ? [team, single] : [st.assistant === 'team' ? team : single];
  }

  function pressStatic(question) {
    const now = pos();
    const key = question ? st.scripted.presses.find((p) => p.label === question) : null;
    const pool = st.scripted.presses.filter((p) => p.after_ms <= now && (!key || p.key === key.key));
    const best = pool.sort((a, b) => b.after_ms - a.after_ms)[0] ||
      { results: ['single_plus', 'team_routed'].map((n) => st.scripted.abstain[n]) };
    const results = chosen().map((a) => best.results.find((r) => r.assistant === a)).filter(Boolean);
    return Promise.resolve({ results, position_ms: now });
  }

  function doPress() {
    if (st.busy) return;
    const question = ui.question.value.trim();
    const now = pos();
    st.busy = true;
    ui.press.classList.add('is-busy');
    nudge(false);
    ui.cards.replaceChildren(el('article', 'rcard rcard--wait', null, [el('p', 'rcard__why', 'Reading what has been said so far' + (live() || st.mode === 'replay' ? '. A real model can take several seconds.' : '…'))]));
    renderSteps();
    const arms = chosen();
    const req = st.mode === 'static'
      ? pressStatic(question)
      : post('api/room/press', { meeting: st.meeting, assistant: arms.length > 1 ? 'both' : arms[0], question: question || null, position_ms: now, session: st.session });
    req.then((r) => {
      showResults(r, 'Uses only what was said before ' + fmt(r.position_ms != null ? r.position_ms : now) + (r.as_of_seq != null && r.as_of_seq >= 0 ? ' (turns up to #' + r.as_of_seq + ')' : ''));
      say('Card ready. The lines it cites are highlighted in the transcript.');
    }).catch((e) => {
      ui.cards.replaceChildren(el('article', 'rcard', null, [el('p', 'rcard__why', 'The press failed: ' + e.message)]));
    }).finally(() => { st.busy = false; ui.press.classList.remove('is-busy'); });
  }
  ui.press.addEventListener('click', doPress);
  ui.nudge.addEventListener('click', doPress);
  ui.bigplay.addEventListener('click', () => { play(); ui.play.focus(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { dismiss(); return; }
    if (e.code !== 'Space' || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest && e.target.closest('input[type="text"], textarea, select, button, summary, a')) return;
    e.preventDefault();
    doPress();
  });

  /* ---- controls ---- */
  ui.assistants.addEventListener('click', (e) => {
    const b = e.target.closest('[data-assistant]');
    if (!b) return;
    st.assistant = b.dataset.assistant;
    ui.assistants.querySelectorAll('[data-assistant]').forEach((n) => n.setAttribute('aria-pressed', String(n === b)));
    if (e.detail) b.blur();
  });
  PRESETS.forEach((q) => {
    const b = el('button', 'ask', q);
    b.type = 'button';
    b.addEventListener('click', () => { ui.question.value = q; ui.question.focus(); });
    ui.presets.append(b);
  });
  ui.play.addEventListener('click', (e) => { if (e.detail) ui.play.blur(); isPlaying() ? pause() : play(); }); /* after a mouse click, space presses rather than toggling playback */
  ui.seek.addEventListener('input', () => seekTo(Number(ui.seek.value) * 1000));
  audio.addEventListener('play', () => { control('play'); syncVideo(); sync(); });
  audio.addEventListener('pause', () => { control('pause'); syncVideo(); sync(); });
  audio.addEventListener('seeked', () => { if (live()) control('seek'); syncVideo(); sync(); });
  audio.addEventListener('ended', () => { say('The meeting has ended.'); sync(); });
  audio.addEventListener('error', () => say('The audio could not be loaded.'));

  function syncVideo() {
    if (ui.video.hidden) return;
    if (Math.abs(ui.video.currentTime - audio.currentTime) > 0.4) ui.video.currentTime = audio.currentTime;
    if (audio.paused) ui.video.pause(); else ui.video.play().catch(() => {});
  }

  function control(action) {
    if (!live() || !st.session) return;
    const body = { action, position_ms: pos(), playing: !audio.paused };
    post('api/room/live/' + st.session + '/control', body).catch((e) => say('Live pipeline: ' + e.message));
  }

  /* ---- loading ---- */
  function resetState() {
    st.turns = []; st.memory = []; st.flags = []; st.cited = new Set(); st.lag = null; st.last = null;
    rendered = { n: 0, last: null }; rows.clear(); memSig = flagSig = '';
    st.started = false; st.carded = false; st.seenFlags = 0; faces.reset(); nudge(false);
    ui.tiles.replaceChildren(); ui.cc.replaceChildren(); ui.cc.dataset.k = '';
    ui.turns.replaceChildren(); ui.cards.replaceChildren(); ui.traces.replaceChildren(); ui.traceWrap.hidden = true;
    ui.capNote.textContent = '';
  }
  function setDuration(ms) {
    st.duration = ms;
    ui.seek.max = String(Math.floor(ms / 1000));
  }

  function loadStatic(data) {
    st.mode = 'static';
    st.scripted = data;
    st.meeting = data.id;
    clock.virtual = true; clock.playing = false; clock.base = 0;
    resetState();
    st.turns = data.turns; st.memory = data.memory; st.flags = data.flags;
    setDuration(data.duration_ms);
    ui.mode.textContent = 'Scripted demo: an invented meeting with prepared cards';
    ui.lead.textContent = 'A short planning meeting plays. Whenever you lose track, press. A card answers with who said it and where.';
    ui.meeting.replaceChildren(new Option(data.title, data.id));
    ui.meeting.hidden = ui.meetingLabel.hidden = true;
    st.arms = ['single_plus', 'team_routed'];
    ui.question.hidden = true;
    ui.question.value = '';
    ui.presets.replaceChildren();
    [...new Set(data.presses.map((p) => p.label))].forEach((q) => {
      const b = el('button', 'ask', q);
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => { ui.question.value = q; ui.presets.querySelectorAll('.ask').forEach((n) => n.setAttribute('aria-pressed', String(n === b))); });
      ui.presets.append(b);
    });
    ui.credit.textContent = data.note;
    say('Ready. Press Play, then press whenever you want a card. Ticks on the timeline mark where something changes.');
    sync();
  }

  function stopLive() {
    clearTimeout(st.pollTimer);
    if (st.session) { fetch('api/room/live/' + st.session, { method: 'DELETE' }).catch(() => {}); st.session = null; }
  }

  function loadMeeting(id, health, meetings) {
    stopLive();
    audio.pause();
    st.meeting = id;
    const m = meetings.find((x) => x.id === id);
    resetState();
    clock.virtual = false;
    setDuration(m.duration_ms);
    audio.src = 'api/room/audio/' + id;
    ui.video.hidden = !m.video;
    if (m.video) { ui.video.src = 'api/room/video/' + id; }
    const load = health.mode === 'live'
      ? post('api/room/live/start', { meeting: id, position_ms: 0 }).then((r) => { st.session = r.session; st.since = 0; poll(); })
      : fetch('api/room/meetings/' + id + '/replay').then((r) => r.json()).then((d) => {
        st.turns = d.turns; st.memory = d.memory; st.flags = d.flags;
        ui.capNote.textContent = d.captions_source;
        ui.memNote.title = d.memory_source;
        ui.credit.textContent = 'Memory: ' + d.memory_source + '. ' + health.attribution;
      });
    load.then(() => say('Ready. Press Play.')).catch((e) => say('Could not load: ' + e.message));
    sync();
  }

  function poll() {
    if (!st.session) return;
    const sid = st.session;
    fetch('api/room/live/' + sid + '/events?since=' + st.since, { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('session ended'))))
      .then((d) => {
        if (sid !== st.session) return;
        st.since = d.next;
        d.events.forEach(applyEvent);
        sync();
        st.pollTimer = setTimeout(poll, 700);
      }).catch(() => { if (sid === st.session) st.pollTimer = setTimeout(poll, 2000); });
  }
  function applyEvent(e) {
    if (e.type === 'reset') { st.turns = []; st.memory = []; st.flags = []; rendered = { n: 0, last: null }; rows.clear(); memSig = flagSig = ''; ui.turns.replaceChildren(); }
    else if (e.type === 'turn') { st.turns.push(e.turn); if (e.speech_lag_ms != null) st.lag = e.speech_lag_ms; }
    else if (e.type === 'turn_update') {
      const i = st.turns.findIndex((t) => t.id === e.turn.id);
      if (i >= 0) { st.turns[i] = e.turn; refreshRow(e.turn); }
    } else if (e.type === 'memory') st.memory.push(e.item);
    else if (e.type === 'flag') st.flags.push(e.flag);
    else if (e.type === 'status' && e.state === 'error') say('Live pipeline: ' + (e.detail || 'error'));
    else if (e.type === 'ended') say('The audio has been sent through the pipeline.');
  }

  function startServer(health) {
    st.mode = health.mode;
    ui.mode.textContent = health.mode === 'live'
      ? 'Live: real speech, speaker labels and local models'
      : 'Replay: a real AMI meeting; cards here are a simple word-overlap stand-in';
    ui.mode.classList.add(health.mode === 'live' ? 'is-live' : 'is-replay');
    ui.lead.textContent = 'A real recorded meeting plays. Whenever you lose track, press. A card answers with who said it and where.';
    if (health.assistants) st.arms = health.assistants;
    return fetch('api/room/meetings').then((r) => r.json()).then((d) => {
      if (!d.meetings.length) throw new Error('no meeting audio found');
      ui.meeting.replaceChildren(...d.meetings.map((m) => new Option(m.id + ' (' + fmt(m.duration_ms) + ')', m.id)));
      ui.meeting.addEventListener('change', () => loadMeeting(ui.meeting.value, health, d.meetings));
      ui.credit.textContent = d.attribution;
      loadMeeting(d.meetings[0].id, health, d.meetings);
    });
  }

  const staticFallback = () => fetch('room-data.json').then((r) => r.json()).then(loadStatic)
    .catch(() => say('The scripted meeting could not be loaded.'));
  if (!window.fetch) { say('This page needs a current browser.'); return; }
  fetch('api/health', { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error('no server'))))
    .then((h) => (h && h.mode ? startServer(h).catch(staticFallback) : staticFallback()))
    .catch(staticFallback);
  window.addEventListener('pagehide', stopLive);
})();
