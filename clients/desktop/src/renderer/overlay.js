import { renderCard } from '../core/card.js';

const $ = (id) => document.getElementById(id);
let hideTimer = null;
let pinned = false;
let asking = false;

function fit() {
  window.cl.invoke('overlay-size', $('frame').getBoundingClientRect().height + 4);
}

function hide() {
  clearTimeout(hideTimer);
  pinned = false;
  asking = false;
  $('pin').classList.remove('on');
  $('ask').hidden = true;
  window.cl.invoke('overlay-hide');
}

function armTimer(seconds) {
  clearTimeout(hideTimer);
  if (!pinned && !asking && seconds) hideTimer = setTimeout(hide, seconds * 1000);
}

function show({ notice, card, detail, question, seconds, ask }) {
  $('notice').hidden = !notice;
  $('notice').textContent = notice ?? '';
  $('ask').hidden = !ask;
  $('card').replaceChildren();
  if (card) {
    if (question) {
      const p = document.createElement('p');
      p.className = 'asked';
      p.textContent = `You asked: ${question}`;
      $('card').append(p);
    }
    $('card').append(renderCard(document, card, { detail }));
  }
  fit();
  armTimer(seconds);
}

window.cl.on('overlay-card', ({ card, detail, question, seconds }) => {
  asking = false;
  show({ card, detail, question, seconds });
});
window.cl.on('overlay-notice', ({ text, kind }) => {
  asking = false;
  show({ notice: text, seconds: kind === 'working' ? 0 : 4 });
  if (kind === 'working') hideTimer = setTimeout(hide, 20000);
});
window.cl.on('overlay-ask', () => {
  asking = true;
  clearTimeout(hideTimer);
  show({ ask: true });
  $('q').value = '';
  $('q').focus();
});

$('ask').addEventListener('submit', (e) => {
  e.preventDefault();
  const q = $('q').value.trim();
  asking = false;
  $('ask').hidden = true;
  if (q) window.cl.invoke('press', q);
  else hide();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') hide();
});
$('close').addEventListener('click', hide);
$('pin').addEventListener('click', () => {
  pinned = !pinned;
  $('pin').classList.toggle('on', pinned);
  if (pinned) clearTimeout(hideTimer);
});
// Click-through when idle: the window only takes the mouse while it is hovered (so Pin and Close work).
$('frame').addEventListener('mouseenter', () => window.cl.invoke('overlay-interactive', true));
$('frame').addEventListener('mouseleave', () => {
  if (!asking) window.cl.invoke('overlay-interactive', false);
});
