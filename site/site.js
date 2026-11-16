(() => {
  document.documentElement.classList.add('js');

  /* Theme preview: ?theme=night|paper|ocean|dusk or the switcher in the corner; remembered per browser. */
  const THEMES = ['night', 'paper', 'ocean', 'dusk'];
  const store = { get() { try { return localStorage.getItem('cl-theme'); } catch (e) { return null; } }, set(v) { try { localStorage.setItem('cl-theme', v); } catch (e) { /* private mode */ } } };
  const applyTheme = (t) => { if (t === 'night' || !THEMES.includes(t)) document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', t); };
  const asked = new URLSearchParams(location.search).get('theme');
  let theme = THEMES.includes(asked) ? asked : (THEMES.includes(store.get()) ? store.get() : 'night');
  applyTheme(theme);
  if (asked) store.set(theme);
  const picker = document.createElement('div');
  picker.className = 'themes';
  picker.setAttribute('role', 'group');
  picker.setAttribute('aria-label', 'Theme');
  picker.append(document.createTextNode('Theme'));
  THEMES.forEach((t) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 't-' + t;
    b.title = t[0].toUpperCase() + t.slice(1);
    b.setAttribute('aria-label', b.title + ' theme');
    b.setAttribute('aria-pressed', String(t === theme));
    b.addEventListener('click', () => {
      theme = t; applyTheme(t); store.set(t);
      picker.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    });
    picker.append(b);
  });
  document.body.append(picker);
  /* Build DOM from data with textContent only, so nothing inserted by script is parsed as HTML. */
  const el = (tag, cls, text, kids) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    (kids || []).forEach((k) => n.append(k));
    return n;
  };
  const row = (term, value, cls) => el('div', null, null, [el('dt', null, term), el('dd', cls || null, value)]);

  document.querySelectorAll('[data-openall]').forEach((b) => {
    b.hidden = false;
    b.addEventListener('click', () => {
      const open = b.getAttribute('aria-pressed') !== 'true';
      document.querySelectorAll('#' + b.dataset.openall + ' details').forEach((d) => { d.open = open; });
      b.setAttribute('aria-pressed', String(open));
      b.textContent = open ? 'Close all' : 'Open all';
    });
  });

  /* a friendly face next to each speaker name in the sample transcripts */
  if (window.CLAvatar) {
    const cast = window.CLAvatar.cast();
    document.querySelectorAll('.turns .who').forEach((w) => {
      if (w.querySelector('.av')) return;
      const name = w.textContent.trim();
      w.classList.add('face-in');
      w.prepend(window.CLAvatar.make(cast.key(name)));
    });
  }

  const card = document.querySelector('[data-card]');
  if (!card) return;
  const states = {
    date: {
      cited: ['t1', 't4'],
      reading: [el('s', 'measured was', 'Friday'), el('span', 'reading', 'Monday')],
      sig: [row('Who', 'Maya', 'person'), row('Where', 't1, t4', 'measured instrument'), row('Read as', 'An update, not a contradiction')]
    },
    pricing: {
      cited: [],
      reading: [el('span', 'reading reading--quiet', 'Not enough information')],
      sig: [row('Who', 'Nobody'), row('Where', 'no turn', 'measured instrument'), row('Read as', 'Pricing never came up, so the card abstains')]
    }
  };
  card.querySelectorAll('[data-ask]').forEach((b) => b.addEventListener('click', () => {
    const s = states[b.dataset.ask];
    card.querySelectorAll('[data-ask]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    card.querySelectorAll('.turns li').forEach((li) => li.classList.toggle('is-cited', s.cited.includes(li.dataset.t)));
    card.querySelector('[data-reading]').replaceChildren(...s.reading);
    card.querySelector('[data-sig]').replaceChildren(...s.sig);
  }));
})();
