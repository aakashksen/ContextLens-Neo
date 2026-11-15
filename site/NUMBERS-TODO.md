# Slots to fill

Numbers: every element with a `data-num` attribute is empty (CSS draws a dash) until it is filled. Set its text to the figure
(for example `0.84`, `1349`, `2832`). The same slot name appears on `index.html` (results band, one per
team and single agent) and on `specs.html` (evaluation table). Fill both places with the same value.

Source: E10, 161 held-out cases from QMSum and AMI, conflicts 20 to 60 turns apart.

| Slot (`data-num`) | What goes in it | Shown on |
|---|---|---|
| `e10-conflict-catch-team` | Conflict catch rate, agent team | index, specs |
| `e10-conflict-catch-single` | Conflict catch rate, single agent | index, specs |
| `e10-cite-team` | Right-turn citation rate, agent team | index, specs |
| `e10-cite-single` | Right-turn citation rate, single agent | index, specs |
| `e10-abstain-team` | Abstention accuracy, agent team | index, specs |
| `e10-abstain-single` | Abstention accuracy, single agent | index, specs |
| `e10-tokens-team` | Tokens per card, agent team | specs |
| `e10-tokens-single` | Tokens per card, single agent | specs |
| `e10-latency-team` | Median latency in ms, agent team | specs |
| `e10-latency-single` | Median latency in ms, single agent | specs |

Pictures: each is a `data-slot` box. To fill one, put an `<img src="img/..." alt="..." width=".." height="..">`
inside it; the placeholder text hides itself and the box takes the picture's own proportions.

| Slot (`data-slot`) | What goes in it | Shown on |
|---|---|---|
| `desktop-shot` | Screenshot of the desktop app, heads-up card over a call (16:10) | index |
| `desktop-shot-main` | The same or a second screenshot of the desktop app | desktop |
| `glasses-shot` | Capture through the glasses or the headset view with the card | index |
| `glasses-shot-main` | The same or a second glasses capture | glasses |

Also pending: the "In development" chip on `desktop.html` and "Headset capture to come" on `glasses.html`
come out once the app and a headset capture exist. The Download buttons point at the install steps on
`desktop.html` until a built installer has a URL.

Checked facts used on the pages that the owner should confirm: the DECODE figures (0.913 against 0.753,
false alerts halved, 45 ms) and the cross-encoder reranker, which `docs/architecture/retrieval-and-memory.md`
on this branch lists as experiment-only.
