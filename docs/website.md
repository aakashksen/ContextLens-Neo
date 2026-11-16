# Website

`site/` is plain HTML, CSS and JavaScript with no framework and no build step. Open `site/index.html` in a browser, or serve the folder with any static host.

## Pages

| File | What it shows |
|---|---|
| `index.html` | What it is in one look, three steps, the two clients, a card to try, the results, under the hood |
| `room.html` | Watch it work: an 80-second invented meeting with avatar seats; cards appear on their own at marked moments, a row of moments jumps to each, Space asks at any point |
| `desktop.html` | The tray app: a hotkey over any call, and that nothing leaves the machine |
| `glasses.html` | The heads-up client: built for display glasses, simulated in Unity today, buildable for headsets like Meta Quest |
| `download.html` | Setup for the desktop app and the glasses client |
| `tour.html` | One invented meeting followed through seven steps of a press, then an abstention and the Unity client |
| `specs.html` | Architecture diagrams, the stack, the experiments and the evaluation |
| `lab.html` | Not linked. The meeting room for developers: a real AMI meeting replayed, or the live pipeline (`tools/meeting_room.py`) |
| `live.html` | Not linked. The earlier scripted replay of two meetings |

## Rules

- Every number comes from `docs/REPORT.md` and the E10/E11 records.
- `room.html` needs no server: the meeting and its cards are scripted in `room-data.json` (`tools/make_room_data.py`). All content is invented.
- Text inserted by script goes through `textContent`, never `innerHTML`. Speaker avatars (`avatars.js`) are constant markup.
- Design tokens are in `tokens.css` (Night is the site theme; Paper, Ocean and Dusk can be previewed with `?theme=`); shared styles are in `site.css`.
- Screenshots: `img/desktop*.webp` are real renders of the Electron app, `img/glasses.webp` is a render of the Unity client (`clients/unity/Assets/Editor/CardCapture.cs`), `img/room-card.webp` is the demo page.
- CSS and JS links carry a `?v=` number; bump it when those files change so browsers fetch the new copy.
