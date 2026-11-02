# ContextLens

Press a button mid-conversation and get one short card based on what was actually said: something you
missed, a possible inconsistency, or help responding. Specialist agents do the digging, and every recalled
fact points back to the words it came from.

This folder holds the whole rebuild: the old code, the design, the plan, and the system that runs the plan.

## Start it

1. Open Claude Code in this folder:
   ```
   cd <local path>
   claude --model sonnet --remote-control
   ```
2. Type `/run-plan 40` to keep it going while you're away. Remote Control lets you answer anything it asks from your phone.

It works through up to 5 tasks per run, stopping whenever it needs you. Type `/run-plan` again to continue,
`/run-plan 10` for a bigger batch, or `/plan-status` to see where things stand. For a hands-off run with a
fresh session for each batch:

```
py -3.12 tools\run_loop.py --max-sessions 10 --tasks-per-session 3
```

It stops on its own when the plan is complete, when it needs you, or after 2 sessions that finish nothing.
Usage limits end it too.

Say yes when Claude Code first asks whether to trust the folder, or its permissions won't load. Use `/run-plan` for tasks that train models, because long runs in the hands-off loop haven't been tested.

## How a task gets finished

```
taskctl next ──> implementer builds it, test first, and commits
                     │
                     ▼
              taskctl verify      reruns the plan's acceptance commands, keeps a log
                     │
                     ▼
              verifier agent      fresh eyes, no build context, tries to break it
                     │
         ┌──── FAIL ─┴─ PASS ────┐
         ▼                       ▼
  back to implementer       taskctl done      refused unless both passed at this exact commit
  (3 tries, then blocked)        │
                                 ▼
                        phase ends? reviewer checks the whole phase
```

## What's where

| Path | What it is |
|---|---|
| `legacy/OfficialContextLens/` | The hackathon code, unchanged |
| `docs/DESIGN.md` | What's being built and why |
| `docs/PLAN.md` | Every task, with files, interfaces, tests |
| `docs/TESTING.md` | Test layers, metrics, experiments |
| `docs/HUMAN-GATES.md` | The parts only you can do |
| `docs/RESUME-TARGETS.md` | Target bullets, and the evidence each number needs |
| `orchestration/tasks.json` | Live task state, changed only through `tools/taskctl.py` |
| `orchestration/verdicts/` | Independent verifier verdicts |
| `results/runs/` | Experiment records, the only source of numbers |
| `results/claims.json` | Numbers allowed onto the resume, each tied to a run |
| `.claude/` | The orchestrator skill, agents and permissions |
