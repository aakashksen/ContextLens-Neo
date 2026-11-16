(() => {
  document.documentElement.classList.add('js');
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
