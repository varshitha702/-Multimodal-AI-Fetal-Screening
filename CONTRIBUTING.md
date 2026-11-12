# Contributing

## Setup

See the quickstart in [`README.md`](README.md). Use Python 3.11 and Node 22.

## Workflow

1. Branch from `main` with a short-lived branch per change: `feat/...`, `fix/...`, `docs/...`, `test/...`,
   `refactor/...`, `chore/...`.
2. Make small commits, one logical change each, using [Conventional Commits](https://www.conventionalcommits.org/):
   `fix(data): fill HC18 ellipse before resize`, `feat(api): add /predict endpoint`, `test(api): ...`, `docs: ...`.
   Suggested scopes: `data`, `train`, `eval`, `api`, `web`, `dashboard`, `test`, `ci`.
3. Before pushing, run:

   ```powershell
   python -m ruff check .
   python -m pytest
   cd frontend; npm run lint; npm run typecheck; npm run build; npm run test:smoke
   ```

4. Open a pull request into `main` with a clear description (what, why, how it was verified). CI must pass.
5. Never force-push `main`, never rewrite history, and never commit datasets, checkpoints (`*.pth`), secrets,
   `.env`, `.venv` or `node_modules`.

## Rules for results

* Only report numbers that were actually produced by the code in this repository, and say where they come from
  (`outputs/*.json`). The test split is used once, for final metrics.
* Keep the "academic research prototype, not a medical device" disclaimer visible in the UI, README and reports.
* Do not add medical formulas (e.g. gestational-age equations) without a cited source and unit tests against
  its published tables.

## Module owners

| Module | Owner |
|---|---|
| Dataset and preprocessing (`ml/masks.py`, `ml/split.py`, `ml/dataset.py`) | Pujari Varshitha |
| Model training and evaluation (`ml/train.py`, `ml/evaluate.py`) | Nachiketha K G |
| FastAPI backend and integration (`backend/`, `ml/inference.py`) | Vishwas K |
| Frontend (`frontend/`) | Rakshitha V |
| Requirements, design docs, testing (`docs/`, `tests/`) | Yashwanth Gowda M N |
