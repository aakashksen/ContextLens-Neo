(() => {
  /* A scripted replay. All content is invented. No network, no server.
     Cards follow docs/client-contract.md: status answered, partial or abstained;
     findings cite only turns spoken before the press. Every string reaches the page through textContent. */
  const root = document.querySelector('[data-demo]');
  if (!root) return;

  const el = (tag, cls, text, kids) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    (kids || []).forEach((k) => n.append(k));
    return n;
  };

  const ABSTAIN = (missing, readAs) => ({
    status: 'abstained',
    reading: { now: 'Not enough information', quiet: true },
    who: 'Nobody',
    readAs: readAs || 'Nothing said so far answers it',
    cites: [],
    findings: [],
    missing: [missing]
  });

  const MEETINGS = [
    {
      id: 'planning',
      title: 'Planning sync',
      turns: [
        ['Maya', 'Let’s ship the beta on Friday.'],
        ['Jordan', 'Design still owes the onboarding screens.'],
        ['Jordan', 'Then QA needs two days, minimum.'],
        ['Maya', 'Okay, make that Monday.'],
        ['Jordan', 'I’ll send the QA plan tomorrow.'],
        ['Maya', 'Priya should see it before we announce.']
      ],
      questions: [
        {
          label: 'When does it ship?',
          rules: [
            { needs: ['t4'], card: { status: 'answered', reading: { was: 'Friday', now: 'Monday' }, who: 'Maya', readAs: 'An update, not a contradiction', cites: ['t1', 't4'], findings: [{ statement: 'Maya moved the beta from Friday to Monday.', cites: ['t1', 't4'] }], suggestion: 'Confirm Monday with design before announcing.' } },
            { needs: ['t1'], card: { status: 'answered', reading: { now: 'Friday' }, who: 'Maya', readAs: 'Said once, nothing has replaced it yet', cites: ['t1'], findings: [{ statement: 'Maya proposed shipping the beta on Friday.', cites: ['t1'] }] } }
          ]
        },
        {
          label: 'What does QA need?',
          rules: [
            { needs: ['t3'], card: { status: 'answered', reading: { now: 'Two days' }, who: 'Jordan', readAs: 'A statement, nothing replaced it', cites: ['t3'], findings: [{ statement: 'Jordan said QA needs two days, at minimum.', cites: ['t3'] }] } }
          ]
        },
        {
          label: 'When is the QA plan due?',
          rules: [
            { needs: ['t5'], card: { status: 'partial', reading: { now: 'Partly answered', quiet: true }, who: 'Jordan', readAs: 'A promise with a relative date', cites: ['t5'], findings: [{ statement: 'Jordan will send the QA plan tomorrow.', cites: ['t5'] }], missing: ['which day “tomorrow” is'] } }
          ]
        },
        { label: 'Who owns pricing?', rules: [], fallback: ABSTAIN('no turn mentions pricing', 'Pricing never came up') }
      ]
    },
    {
      id: 'budget',
      title: 'Budget check-in',
      turns: [
        ['Ana', 'Let us move the launch to Friday.'],
        ['Ben', 'Friday works, but I need the budget sheet first.'],
        ['Ana', 'I will send the budget sheet tomorrow morning.'],
        ['Ben', 'Actually, make that Monday, I am out tomorrow.']
      ],
      questions: [
        {
          label: 'When is the launch?',
          rules: [
            { needs: ['t2'], card: { status: 'answered', reading: { now: 'Friday' }, who: 'Ana', readAs: 'Proposed, and agreed on a condition', cites: ['t1', 't2'], findings: [{ statement: 'Ana proposed Friday. Ben said it works once he has the budget sheet.', cites: ['t1', 't2'] }], suggestion: 'Ask Ana whether the sheet is ready.' } },
            { needs: ['t1'], card: { status: 'answered', reading: { now: 'Friday' }, who: 'Ana', readAs: 'Proposed, no reply yet', cites: ['t1'], findings: [{ statement: 'Ana proposed moving the launch to Friday.', cites: ['t1'] }] } }
          ]
        },
        {
          label: 'When does the sheet arrive?',
          rules: [
            { needs: ['t4'], card: { status: 'answered', reading: { was: 'Tomorrow morning', now: 'Monday' }, who: 'Ben', readAs: 'An update, not a contradiction', cites: ['t3', 't4'], findings: [{ statement: 'Ana offered tomorrow morning. Ben asked to make it Monday because he is out tomorrow.', cites: ['t3', 't4'] }] } },
            { needs: ['t3'], card: { status: 'answered', reading: { now: 'Tomorrow morning' }, who: 'Ana', readAs: 'A promise, nothing has changed it yet', cites: ['t3'], findings: [{ statement: 'Ana will send the budget sheet tomorrow morning.', cites: ['t3'] }] } }
          ]
        },
        { label: 'What is the budget total?', rules: [], fallback: ABSTAIN('no turn gives a budget figure', 'No figure was ever said') }
      ]
    }
  ];

  window.CL_MEETINGS = MEETINGS; // read by live-local.js when a local server answers

  root.querySelectorAll('[hidden]').forEach((n) => { n.hidden = false; });

  const STEP_MS = 2200;
  const LABEL_MS = 800;

  const tabs = root.querySelector('[data-tabs]');
  const turnsEl = root.querySelector('[data-turns]');
  const asksEl = root.querySelector('[data-asks]');
  const cardEl = root.querySelector('[data-card]');
  const replayBtn = root.querySelector('[data-replay]');
  const skipBtn = root.querySelector('[data-skip]');
  const statusEl = root.querySelector('[data-status]');

  let meeting = MEETINGS[0];
  let shown = 0;
  let timers = [];
  let run = 0;

  const clearTimers = () => { timers.forEach(clearTimeout); timers = []; };
  const idOf = (i) => 't' + (i + 1);
  const visibleIds = () => meeting.turns.slice(0, shown).map((_, i) => idOf(i));

  const turnRow = (i, speakerKnown) => {
    const [who, text] = meeting.turns[i];
    const li = el('li', null, null, [
      el('span', 'measured id', idOf(i)),
      el('span', speakerKnown ? 'who' : 'who who--unknown', speakerKnown ? who : 'unknown'),
      el('span', null, text)
    ]);
    li.dataset.t = idOf(i);
    return li;
  };

  const setStatus = (text) => { statusEl.textContent = text; };

  const addTurn = (i, instant) => {
    const li = turnRow(i, instant);
    turnsEl.append(li);
    shown = i + 1;
    if (!instant) {
      timers.push(setTimeout(() => {
        const labelled = turnRow(i, true);
        if (li.classList.contains('is-cited')) labelled.classList.add('is-cited');
        li.replaceWith(labelled);
      }, LABEL_MS));
    }
    setStatus(shown < meeting.turns.length ? 'Recording' : 'Replay finished');
  };

  const clearCard = () => {
    cardEl.replaceChildren(el('p', 'empty', shown ? 'Press a question. The card cites only what was said so far.' : 'Waiting for the first turn.'));
    turnsEl.querySelectorAll('li').forEach((li) => li.classList.remove('is-cited'));
    asksEl.querySelectorAll('.ask').forEach((b) => b.setAttribute('aria-pressed', 'false'));
  };

  const start = (instant) => {
    clearTimers();
    run += 1;
    shown = 0;
    turnsEl.replaceChildren();
    clearCard();
    setStatus('Recording');
    if (instant) {
      meeting.turns.forEach((_, i) => addTurn(i, true));
      clearCard();
      return;
    }
    meeting.turns.forEach((_, i) => {
      timers.push(setTimeout(() => { addTurn(i, false); if (i === 0) clearCard(); }, 600 + i * STEP_MS));
    });
  };

  const pick = (q) => {
    const have = new Set(visibleIds());
    const hit = q.rules.find((r) => r.needs.every((n) => have.has(n)));
    return hit ? hit.card : (q.fallback || ABSTAIN('nothing said so far covers it'));
  };

  const quoteLine = (id) => {
    const i = Number(id.slice(1)) - 1;
    const [who, text] = meeting.turns[i];
    return el('p', 'quote', null, [el('b', 'measured', id), document.createTextNode(' ' + who + ': “' + text + '”')]);
  };

  const buildCard = (c, pressedAfter) => {
    const chipCls = c.status === 'answered' ? 'chip chip--ok' : c.status === 'partial' ? 'chip chip--person' : 'chip';
    const reading = el('div', null, null, []);
    if (c.reading.was) reading.append(el('s', 'measured was', c.reading.was));
    reading.append(el('span', c.reading.quiet ? 'reading reading--quiet' : 'reading', c.reading.now));
    const sigRow = (term, value, cls) => el('div', null, null, [el('dt', null, term), el('dd', cls || null, value)]);
    const sig = el('dl', 'sig', null, [
      sigRow('Who', c.who, c.who === 'Nobody' ? null : 'person'),
      sigRow('Where', c.cites.length ? c.cites.join(', ') : 'no turn', 'measured instrument'),
      sigRow('Read as', c.readAs)
    ]);
    const card = el('div', 'card', null, [
      el('div', 'card__top', null, [el('span', chipCls, c.status), el('span', 'measured', 'pressed after ' + pressedAfter)]),
      el('div', 'answer', null, [reading, sig])
    ]);
    if (c.findings.length) {
      card.append(el('ul', 'finds', null, c.findings.map((f) => el('li', null, null, [el('p', null, f.statement)].concat(f.cites.map(quoteLine))))));
    }
    const more = [];
    if (c.missing && c.missing.length) more.push(el('p', null, null, [el('b', null, 'Missing: '), document.createTextNode(c.missing.join('; '))]));
    if (c.suggestion) more.push(el('p', null, null, [el('b', null, 'Suggestion: '), document.createTextNode(c.suggestion)]));
    if (more.length) card.append(el('div', 'card__more', null, more));
    return card;
  };

  const press = (q, button) => {
    if (!shown) {
      cardEl.replaceChildren(el('p', 'empty', 'Nothing heard yet. No card is made.'));
      return;
    }
    const c = pick(q);
    asksEl.querySelectorAll('.ask').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    turnsEl.querySelectorAll('li').forEach((li) => li.classList.toggle('is-cited', c.cites.includes(li.dataset.t)));
    cardEl.replaceChildren(buildCard(c, idOf(shown - 1)));
  };

  const renderAsks = () => {
    asksEl.replaceChildren(...meeting.questions.map((q) => {
      const b = el('button', 'ask', q.label);
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => press(q, b));
      return b;
    }));
  };

  const renderTabs = () => {
    tabs.replaceChildren(...MEETINGS.map((m) => {
      const b = el('button', 'ask', m.title);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(m === meeting));
      b.addEventListener('click', () => { meeting = m; renderTabs(); renderAsks(); start(false); });
      return b;
    }));
  };

  replayBtn.addEventListener('click', () => start(false));
  skipBtn.addEventListener('click', () => start(true));

  renderTabs();
  renderAsks();
  start(false);
})();
