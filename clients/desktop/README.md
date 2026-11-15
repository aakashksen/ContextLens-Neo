# ContextLens desktop

A tray app for using ContextLens in a real meeting. It listens (microphone, and on Windows the other side of a call), streams audio to the local backend over `/ws/{conversation_id}`, and when you press a hotkey shows a small heads-up card in a screen corner: what was actually said, with the quoted turn and speaker.

**Privacy.** Everything is local. Audio and transcripts go only to the backend on this machine (`ws://127.0.0.1:8000` by default). The app has no telemetry, no crash reporting and no auto-update, and it keeps no transcript on disk: its settings file holds only your settings and the current conversation id.

## Run it

Needs Node 22 or newer. From this folder:

```
npm install
npm run dev        # run the app from source
npm run dist       # build the Windows installer: dist/ContextLens Setup 0.1.0.exe
npm test           # unit tests, fake-server end-to-end test, Electron smoke test
```

`dist/` is ignored by git. The installer is unsigned, so Windows SmartScreen may ask you to confirm.

## First run

1. Start Ollama yourself (`ollama serve`), with the two models pulled.
2. Open ContextLens from the tray. The **Backend** panel lists Ollama models, the speech service, PostgreSQL and the API, each as ready, missing or warming, and for anything missing it shows the one command to fix it (with a Copy button):
   - `ollama pull qwen3.5:4b-q4_K_M` and `ollama pull nomic-embed-text`
   - `docker compose up -d speech`
   - `docker compose up -d db`
3. Press **Start backend**. This runs `python -m contextlens.api.production` from the repo's `.venv` with `CONTEXTLENS_ASSISTANT` set from Settings. The app owns that process and stops it when you stop it or quit. The API is "warming" until both models are loaded, then "ready".
   - Run from the repo (`npm run dev`) and the repo is found automatically. For an installed app, set **ContextLens folder** (the folder that contains `src/contextlens`) in Settings. `DATABASE_URL`, `SPEECH_ADDR` and `HF_TOKEN` come from your own environment, as in `docs/run-everything.md`.
   - If you start the backend yourself, the app just connects to it.
4. The status pill shows **Listening** once connected. Talk; the transcript tab fills in.

## The press

| | |
|---|---|
| `Ctrl+Shift+Space` | Show the card for what was said so far |
| `Ctrl+Shift+Q` | Open a small box, type a question, press Enter |
| Tray menu | The same two, plus pause/resume, forget, start/stop backend, quit |

Both hotkeys are configurable in Settings (a modifier plus one key, for example `Ctrl+Alt+F9`). If another app owns a hotkey the main window says so.

The card appears in the corner you choose (always on top, frameless). While idle it ignores the mouse, so it never blocks what is under it; hover it to use **Pin** (keep it) or **Close**. Otherwise it hides after the number of seconds in Settings. It shows the status (Answered, Partial answer, Not enough information), each finding with its quoted turn, speaker and time, and the conflict label when there is one ("Conflicts with what was said earlier", "Changes what was said earlier").

Detail level (Settings): **brief** shows the first finding only; **standard** adds the quotes; **detailed** adds the suggestion, gaps and timing. It is also sent as the press's `detail`.

## Audio sources

Settings has two switches: **Microphone** (default on) and **System audio** (Windows loopback, default off). Switched on together they are mixed into one stream, so both sides of a call are heard. The tray icon shows a red dot while audio is being captured. **Pause** stops capture and tells the backend to ignore audio; **Forget this conversation** sends `delete`, clears the transcript, memory and cards held in the app, and starts a new conversation id.

## Settings

Hotkeys, audio sources, assistant (default `single_plus`), detail, auto-hide seconds, overlay corner, server URL, repo folder and Python path. The assistant is read by the backend when it starts, so after changing it press Stop backend, then Start backend. Settings live in `settings.json` in the app's user-data folder (under `%APPDATA%`).

## Main window

Transcript with speakers (`unknown` until the backend labels a turn), Memory items (shown when the backend sends them), Past cards, and Settings. Select a past card to see **How it got here**: stage timings, which specialist ran, the checker verdict and what the guards dropped. The backend does not send that trace today; the panel reads an optional `trace` object on the card and says so when it is absent. See `docs/architecture/desktop-client.md` for the shape.
