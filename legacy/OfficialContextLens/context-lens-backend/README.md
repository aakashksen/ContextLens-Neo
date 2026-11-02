Context Lens Backend (Scaffold)

Overview
- Real-time game-engine style backend that emits one HUD Frame JSON per line to stdout
- Logs are written only to stderr
- Sim stubs for audio, vision, and brain (no external APIs)

Run
- npm run start
- npm run dev

Windows note for dev env vars
- Powershell:
  $env:SIM_AUDIO='1'; $env:SIM_BRAIN='1'; $env:SIM_VISION='1'; node index.js
- cmd.exe:
  set SIM_AUDIO=1 && set SIM_BRAIN=1 && set SIM_VISION=1 && node index.js

Sim flags
- SIM_AUDIO=1 to inject mock audio events
- SIM_VISION=1 to inject mock vision summaries
- SIM_BRAIN=1 to synthesize Brain UPDATE/SKIP decisions

What to expect
- stdout: one HUD Frame JSON object per line at the hud_tick_ms interval
- stderr: operational logs

Intentionally stubbed
- Audio capture, vision processing, and external LLM calls
