VERDICT: PASS

Phase 0 (T0.1 to T0.3) was reviewed at HEAD 8496c48 with a clean working tree.

Checks run:
- `.venv/Scripts/python -m pytest -q`: 23 passed. `ruff check .` is clean.
- `pyproject.toml` has `pythonpath`, `addopts = "-m 'not real'"`, the `real` marker, and a ruff exclude for `legacy`.
- The compose file uses `pgvector/pgvector:pg16`. `.env.example` holds placeholders only, and `.env` is git-ignored.
- CI runs ruff and pytest.
- `src/contextlens/schemas.py` matches the PLAN.md T0.2 interface in names, types, defaults and Literals. Card and Finding invariants are real validators, and the tests exercise them.
- `docs/datasets.md` has URL, License, Labels and Approved for all four datasets. The model candidates have Approved lines too. The checker passes.
- `results/` holds no numbers, and `docs/datasets.md` states no metrics. The README and docs make no unearned claims. No `legacy/` edits, no skipped tests, and no broad exception handling.
- No leakage surface exists yet, since there is no retrieval code.

No concrete defects.

Observations (not defects):
1. `docs/datasets.md` marks DECODE and QMSum `Approved: no, needs the person`. T1.1 (`load_pairs`, `load_gold`) and the later consistency training (Phase 5) use both. PLAN.md says H1 "doesn't block the build; blocks T9.1". In practice T1.1 can only download LongMemEval and AMI until the person decides, so the orchestrator should raise this at T1.1 rather than skip the DECODE and QMSum loaders silently.
2. Ruff selects only E4, E7, E9 and F. This is a light lint, but the plan does not require more.
3. No test checks that non-recall findings may have empty evidence. This is a minor gap, and the plan does not ask for it.
