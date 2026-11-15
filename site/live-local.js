(() => {
  /* Live mode. Runs only when a local server (tools/live_demo.py) answers /api/health; otherwise the scripted replay
     in live.js stays as it is. Cards come from the real assistant on a local model. Every string reaches the page
     through textContent. */
  const root = document.querySelector('[data-demo]');
  const meetings = window.CL_MEETINGS;
  if (!root || !meetings || !window.fetch) return;

  const el = (tag, cls, text, kids) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    (kids || []).forEach((k) => n.append(k));
    return n;
  };
  const button = (label, cls, onClick) => {
    const b = el('button', cls, label);
    b.type = 'button';
    b.addEventListener('click', onClick);
    return b;
  };

  fetch('api/health', { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error('no server'))))
    .then((health) => start(health))
    .catch(() => {});

  const KINDS = [['single', 'Single agent'], ['team', 'Agent team'], ['both', 'Both']];
  const NAMES = { single: 'Single agent', team: 'Agent team' };

  function start(health) {
    let meeting = meetings[0];
    let rows = meeting.turns.map(([speaker, text]) => ({ speaker, text }));
    let question = meeting.questions[0].label;
    let kind = 'both';
    let busy = false;

    const hero = root.closest('.hero__grid');
    hero.classList.add('is-live');
    hero.querySelector('.lead').textContent = 'The real assistant, running on a local model on this machine. Pick or edit a meeting, ask a question, press.';
    hero.querySelector('h1').before(el('span', 'chip chip--ok livebadge', 'Live: local model, ' + health.model));

    const meetingBar = el('div', 'asks');
    meetingBar.setAttribute('role', 'group');
    meetingBar.setAttribute('aria-label', 'Meeting');
    const turnsEl = el('ol', 'turns turns--edit');
    const asksEl = el('div', 'asks');
    asksEl.setAttribute('role', 'group');
    asksEl.setAttribute('aria-label', 'Preset questions');
    const qInput = el('input', 'field');
    qInput.type = 'text';
    qInput.maxLength = 300;
    qInput.placeholder = 'Type a question, or leave empty for a general press';
    qInput.setAttribute('aria-label', 'Question');
    qInput.addEventListener('input', () => { question = qInput.value; markAsks(); });
    const kindBar = el('div', 'asks');
    kindBar.setAttribute('role', 'group');
    kindBar.setAttribute('aria-label', 'Assistant');
    const pressBtn = button('Press', 'btn btn--primary', () => press());
    const resultEl = el('div', null, null, [el('p', 'empty', 'Press to get a card from the local model.')]);
    resultEl.setAttribute('aria-live', 'polite');

    const markAsks = () => asksEl.querySelectorAll('.ask').forEach((b) => b.setAttribute('aria-pressed', String(b.textContent === question)));

    const renderAsks = () => {
      asksEl.replaceChildren(...meeting.questions.map((q) => {
        const b = button(q.label, 'ask', () => { question = q.label; qInput.value = q.label; markAsks(); });
        b.setAttribute('aria-pressed', String(q.label === question));
        return b;
      }));
    };

    const renderRows = () => {
      turnsEl.replaceChildren(...rows.map((r, i) => {
        const who = el('input', 'field field--who');
        who.value = r.speaker; who.maxLength = 40; who.placeholder = 'Speaker';
        who.setAttribute('aria-label', 'Speaker of turn ' + (i + 1));
        who.addEventListener('input', () => { r.speaker = who.value; });
        const text = el('input', 'field field--text');
        text.value = r.text; text.maxLength = 600; text.placeholder = 'What was said';
        text.setAttribute('aria-label', 'Text of turn ' + (i + 1));
        text.addEventListener('input', () => { r.text = text.value; });
        const rm = button('Remove', 'ask', () => { rows.splice(i, 1); renderRows(); });
        rm.setAttribute('aria-label', 'Remove turn ' + (i + 1));
        const li = el('li', null, null, [el('span', 'measured id', 't' + (i + 1)), who, text, rm]);
        li.dataset.t = 't' + (i + 1);
        return li;
      }));
      const add = button('Add turn', 'ask', () => {
        rows.push({ speaker: '', text: '' });
        renderRows();
        [...turnsEl.querySelectorAll('li[data-t]')].pop().querySelector('input').focus();
      });
      turnsEl.append(el('li', 'turns__add', null, [add]));
    };

    const renderMeetings = () => {
      meetingBar.replaceChildren(...meetings.map((m) => {
        const b = button(m.title, 'ask', () => {
          meeting = m;
          rows = m.turns.map(([speaker, text]) => ({ speaker, text }));
          question = m.questions[0].label;
          qInput.value = question;
          renderMeetings(); renderRows(); renderAsks();
        });
        b.setAttribute('aria-pressed', String(m === meeting));
        return b;
      }));
    };

    const renderKinds = () => {
      kindBar.replaceChildren(...KINDS.map(([k, label]) => {
        const b = button(label, 'ask', () => { kind = k; renderKinds(); });
        b.setAttribute('aria-pressed', String(k === kind));
        return b;
      }));
    };

    const quote = (e) => el('p', 'quote', null, [el('b', 'measured', e.turn_id), document.createTextNode(' ' + e.speaker + ': “' + e.quote + '”')]);

    const buildCard = (res) => {
      const c = res.card;
      const chipCls = c.status === 'answered' ? 'chip chip--ok' : c.status === 'partial' ? 'chip chip--person' : 'chip';
      const card = el('div', 'card', null, [el('div', 'card__top', null, [
        el('span', chipCls, c.status),
        el('span', 'chip', c.mode),
        el('span', 'measured', (res.latency_ms / 1000).toFixed(1) + ' s · ' + res.tokens + ' tokens')
      ])]);
      if (!c.findings.length) {
        card.append(el('div', 'answer', null, [el('span', 'reading reading--quiet', 'Not enough information')]));
      } else {
        card.append(el('ul', 'finds', null, c.findings.map((f) => el('li', null, null,
          [el('p', null, f.statement)].concat(f.evidence.map(quote), [el('span', 'measured', f.agent + ' · ' + f.kind)])))));
      }
      const more = [];
      if (c.missing && c.missing.length) more.push(el('p', null, null, [el('b', null, 'Missing: '), document.createTextNode(c.missing.join('; '))]));
      if (c.suggestion) more.push(el('p', null, null, [el('b', null, 'Suggestion: '), document.createTextNode(c.suggestion)]));
      if (more.length) card.append(el('div', 'card__more', null, more));
      return card;
    };

    const press = () => {
      if (busy) return;
      const turns = rows.map((r) => ({ speaker: r.speaker.trim(), text: r.text.trim() })).filter((r) => r.speaker || r.text);
      if (!turns.length || turns.some((t) => !t.speaker || !t.text)) {
        resultEl.replaceChildren(el('p', 'empty', 'Every turn needs a speaker and text.'));
        return;
      }
      const kinds = kind === 'both' ? ['single', 'team'] : [kind];
      busy = true;
      pressBtn.disabled = true;
      turnsEl.querySelectorAll('li').forEach((li) => li.classList.remove('is-cited'));
      const cols = kinds.map((k) => el('div', 'demo__col', null, [el('div', 'demo__cap', NAMES[k]), el('p', 'empty', 'Running on the local model…')]));
      const grid = el('div', 'livecols', null, cols);
      grid.style.setProperty('--cols', String(kinds.length));
      resultEl.replaceChildren(grid);
      const cited = new Set();
      Promise.all(kinds.map((k, i) =>
        fetch('api/press', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ turns, question: question.trim() || null, assistant: k }) })
          .then((r) => r.json().then((j) => (r.ok ? j : Promise.reject(new Error(j.error || 'request failed')))))
          .then((res) => {
            cols[i].lastChild.replaceWith(buildCard(res));
            res.card.findings.forEach((f) => f.evidence.forEach((e) => cited.add(e.turn_id)));
            turnsEl.querySelectorAll('li[data-t]').forEach((li) => li.classList.toggle('is-cited', cited.has(li.dataset.t)));
          })
          .catch((err) => { cols[i].lastChild.replaceWith(el('p', 'empty', 'No card: ' + err.message)); })
      )).finally(() => { busy = false; pressBtn.disabled = false; });
    };

    qInput.value = question;
    renderMeetings(); renderRows(); renderAsks(); renderKinds();
    root.replaceChildren(
      el('div', 'demo__bar', null, [meetingBar]),
      el('div', 'demo__cap', 'Transcript ', [el('span', null, 'edit any turn')]),
      turnsEl,
      el('div', 'demo__cap', 'Question'),
      el('div', 'livebar', null, [qInput]),
      el('div', 'livebar', null, [asksEl]),
      el('div', 'demo__cap', 'Assistant'),
      el('div', 'livebar', null, [kindBar, pressBtn]),
      resultEl
    );
  }
})();
