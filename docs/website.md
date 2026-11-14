# Website

`site/` is plain HTML, CSS and JavaScript with no framework and no build step. Open `site/index.html` in a browser, or serve the folder with any static host.

## Pages

| File | What it shows |
|---|---|
| `index.html` | The question, a working card, how a press works, the team against one agent, what broke, the stack, contact |
| `tour.html` | One invented meeting followed through seven steps, then an abstention and the Unity client |
| `specs.html` | Architecture diagrams, the stack, experiments E1 to E8 with the rule written first and the failures, the limits |
| `live.html` | A scripted replay of two invented meetings. Press a question and get a card |

## Rules

- Every number comes from `docs/REPORT.md`. Where the report withholds a figure (groundedness), the page says it is not published.
- `live.html` needs no server. The meetings and cards are scripted in `live.js`, shaped after `tools/fake_server.py` and `docs/client-contract.md`. All content is invented.
- Text inserted by script goes through `textContent`, never `innerHTML`.
- Design tokens are in `tokens.css`; shared styles are in `site.css`.
