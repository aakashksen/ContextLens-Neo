/* Friendly speaker avatars: a small cast of robots and animals, drawn inline.
   CLAvatar.make(key) returns an <svg>; CLAvatar.cast() gives each new speaker the next unused character,
   so the people in one meeting never share a face. The markup is constant, never built from page data. */
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const D = '#1e2533';
  const CAST = [
    ['robot', '#2f4f86', `<path d="M20 64v-8a12 9 0 0 1 24 0v8z" fill="#8fa3c2"/><rect x="17" y="21" width="30" height="25" rx="8" fill="#d5deec"/><path d="M32 21v-7" stroke="#d5deec" stroke-width="2.5"/><circle cx="32" cy="12" r="3.5" fill="#ffc94a"/><rect x="13" y="29" width="4" height="10" rx="2" fill="#8fa3c2"/><rect x="47" y="29" width="4" height="10" rx="2" fill="#8fa3c2"/><rect x="21" y="26" width="22" height="12" rx="6" fill="${D}"/><g class="av-eyes" fill="#6fe3ff"><circle cx="27" cy="32" r="2.6"/><circle cx="37" cy="32" r="2.6"/></g><rect class="av-mouth" x="27" y="40.5" width="10" height="2.2" rx="1.1" fill="${D}"/>`],
    ['fox', '#245a4c', `<path d="M15 27 16 9l12 11zM49 27 48 9 36 20z" fill="#f08a3c"/><path d="M18 24 18.5 14l6 6zM46 24 45.5 14l-6 6z" fill="#7a3a16"/><path d="M13 27q3-9 19-8 16-1 19 8L32 52z" fill="#f08a3c"/><path d="M15 29q11 1 17 23-14-8-17-23zM49 29q-11 1-17 23 14-8 17-23z" fill="#fff4e8"/><g class="av-eyes" fill="${D}"><circle cx="25.5" cy="30" r="2.3"/><circle cx="38.5" cy="30" r="2.3"/></g><circle cx="32" cy="48.5" r="2.4" fill="${D}"/><path class="av-mouth" d="M30 44h4" stroke="${D}" stroke-width="1.4" stroke-linecap="round"/>`],
    ['cat', '#5b3f86', `<path d="M17 31 19 11l12 10zM47 31 45 11 33 21z" fill="#f2a65a"/><path d="M20 24l1.2-8 6 5zM44 24l-1.2-8-6 5z" fill="#f7c9a0"/><circle cx="32" cy="37" r="16" fill="#f2a65a"/><path d="M28 22q4 3 8 0" stroke="#d9823a" stroke-width="2" fill="none"/><g class="av-eyes" fill="${D}"><ellipse cx="25.5" cy="35" rx="2.2" ry="3"/><ellipse cx="38.5" cy="35" rx="2.2" ry="3"/></g><path d="M30.5 40h3L32 42z" fill="#d9605a"/><path class="av-mouth" d="M29 44q1.5 1.6 3 0 1.5 1.6 3 0" stroke="${D}" stroke-width="1.2" fill="none"/><path d="M14 39l8 1M14 43l8-1M50 39l-8 1M50 43l-8-1" stroke="#fff4e8" stroke-width="1" opacity=".8"/>`],
    ['panda', '#2e6b5c', `<circle cx="19" cy="22" r="6.5" fill="#262a33"/><circle cx="45" cy="22" r="6.5" fill="#262a33"/><circle cx="32" cy="36" r="17" fill="#f4f4f0"/><ellipse cx="25" cy="34" rx="4.6" ry="5.8" transform="rotate(-25 25 34)" fill="#262a33"/><ellipse cx="39" cy="34" rx="4.6" ry="5.8" transform="rotate(25 39 34)" fill="#262a33"/><g class="av-eyes" fill="#fff"><circle cx="25.6" cy="33.4" r="1.5"/><circle cx="38.4" cy="33.4" r="1.5"/></g><ellipse cx="32" cy="40.5" rx="2.6" ry="1.8" fill="#262a33"/><path class="av-mouth" d="M29.5 44q2.5 2 2.5 0 0 2 2.5 0" stroke="#262a33" stroke-width="1.3" fill="none"/>`],
    ['owl', '#7a3f55', `<path d="M17 25 18 13l8 8zM47 25 46 13l-8 8z" fill="#9b7653"/><ellipse cx="32" cy="38" rx="17" ry="18" fill="#9b7653"/><ellipse cx="32" cy="47" rx="9" ry="8" fill="#c9a77f"/><circle cx="25" cy="33" r="7.2" fill="#f3e6cf"/><circle cx="39" cy="33" r="7.2" fill="#f3e6cf"/><g class="av-eyes" fill="${D}"><circle cx="25" cy="33" r="3.3"/><circle cx="39" cy="33" r="3.3"/></g><circle cx="26" cy="32" r="1" fill="#fff"/><circle cx="40" cy="32" r="1" fill="#fff"/><path class="av-mouth" d="M30 38.5h4L32 43z" fill="#f2b33d"/>`],
    ['frog', '#2f4f7a', `<circle cx="22" cy="25" r="7.5" fill="#6cc070"/><circle cx="42" cy="25" r="7.5" fill="#6cc070"/><ellipse cx="32" cy="39" rx="19" ry="14" fill="#6cc070"/><circle cx="22" cy="25" r="4.6" fill="#fff"/><circle cx="42" cy="25" r="4.6" fill="#fff"/><g class="av-eyes" fill="${D}"><circle cx="22.5" cy="25.5" r="2.4"/><circle cx="41.5" cy="25.5" r="2.4"/></g><circle cx="19" cy="41" r="2.6" fill="#f29a9a" opacity=".7"/><circle cx="45" cy="41" r="2.6" fill="#f29a9a" opacity=".7"/><path class="av-mouth" d="M23 40q9 8 18 0" stroke="#2d5a2f" stroke-width="2" fill="none" stroke-linecap="round"/>`],
    ['bot', '#7a4535', `<path d="M21 64v-7a11 8 0 0 1 22 0v7z" fill="#c9ced8"/><path d="M15 44a17 17 0 0 1 34 0v3H15z" fill="#e6ebf2"/><path d="M24 23l-4-8M40 23l4-8" stroke="#e6ebf2" stroke-width="2.2"/><circle cx="20" cy="14" r="2.4" fill="#ff7a6b"/><circle cx="44" cy="14" r="2.4" fill="#ff7a6b"/><rect x="19" y="30" width="26" height="9" rx="4.5" fill="${D}"/><g class="av-eyes"><rect x="25" y="32.5" width="14" height="4" rx="2" fill="#ff8f7f"/></g><rect class="av-mouth" x="28" y="42" width="8" height="2" rx="1" fill="#9aa3b2"/>`],
    ['penguin', '#3c6a8c', `<ellipse cx="32" cy="37" rx="17" ry="18" fill="#263040"/><path d="M32 26c-10-4-14 8-10 16 3 7 17 7 20 0 4-8 0-20-10-16z" fill="#f4f6f8"/><g class="av-eyes" fill="${D}"><circle cx="27" cy="34" r="2.2"/><circle cx="37" cy="34" r="2.2"/></g><path class="av-mouth" d="M28.5 38h7L32 42.5z" fill="#f59a3c"/><circle cx="24" cy="40" r="2" fill="#f29a9a" opacity=".6"/><circle cx="40" cy="40" r="2" fill="#f29a9a" opacity=".6"/>`],
    ['bunny', '#5a3f7a', `<ellipse cx="25" cy="16" rx="4.8" ry="12" fill="#ece6f0"/><ellipse cx="39" cy="16" rx="4.8" ry="12" fill="#ece6f0"/><ellipse cx="25" cy="16" rx="2.2" ry="8" fill="#f2b8c8"/><ellipse cx="39" cy="16" rx="2.2" ry="8" fill="#f2b8c8"/><circle cx="32" cy="39" r="15" fill="#ece6f0"/><g class="av-eyes" fill="${D}"><circle cx="26.5" cy="37" r="2.2"/><circle cx="37.5" cy="37" r="2.2"/></g><path d="M30.6 41.5h2.8L32 43.2z" fill="#e57a92"/><path class="av-mouth" d="M30.5 45h3v2.5h-3z" fill="#fff" stroke="#c9bfd0" stroke-width=".6"/>`],
    ['bear', '#3f5f7a', `<circle cx="19" cy="22" r="6.5" fill="#a8754a"/><circle cx="45" cy="22" r="6.5" fill="#a8754a"/><circle cx="19" cy="22" r="3.2" fill="#d6a77a"/><circle cx="45" cy="22" r="3.2" fill="#d6a77a"/><circle cx="32" cy="37" r="17" fill="#a8754a"/><ellipse cx="32" cy="43" rx="7.5" ry="6" fill="#e3c49e"/><g class="av-eyes" fill="${D}"><circle cx="25.5" cy="33.5" r="2.2"/><circle cx="38.5" cy="33.5" r="2.2"/></g><ellipse cx="32" cy="40.5" rx="2.9" ry="2" fill="#3a2616"/><path class="av-mouth" d="M29.5 45q2.5 2 2.5 0 0 2 2.5 0" stroke="#3a2616" stroke-width="1.3" fill="none"/>`]
  ];
  const UNKNOWN = ['someone', '#3a4150', `<circle cx="32" cy="27" r="10" fill="#8a93a3"/><path d="M14 58a18 15 0 0 1 36 0z" fill="#8a93a3"/>`];

  function make(key) {
    const c = CAST.find((x) => x[0] === key) || UNKNOWN;
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 64 64');
    svg.setAttribute('class', 'av av--' + c[0]);
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    const doc = new DOMParser().parseFromString(`<svg xmlns="${NS}"><circle cx="32" cy="32" r="32" fill="${c[1]}"/>${c[2]}</svg>`, 'image/svg+xml');
    [...doc.documentElement.childNodes].forEach((n) => svg.append(document.importNode(n, true)));
    return svg;
  }
  const hash = (s) => [...String(s)].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) >>> 0, 7);

  /* One cast per meeting: first come, first served, starting from a stable spot for the first name. */
  function cast() {
    const given = new Map();
    return {
      key(name) {
        if (!name || name === 'unknown') return 'someone';
        if (given.has(name)) return given.get(name);
        const used = new Set(given.values());
        const start = given.size ? 0 : hash(name) % CAST.length;
        let k = CAST[start][0];
        for (let i = 0; i < CAST.length; i++) { const c = CAST[(start + i) % CAST.length][0]; if (!used.has(c)) { k = c; break; } }
        given.set(name, k);
        return k;
      },
      reset() { given.clear(); }
    };
  }
  window.CLAvatar = { make, cast, keys: CAST.map((c) => c[0]) };
})();
