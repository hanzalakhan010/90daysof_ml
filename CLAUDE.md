# 90daysof_ml

This is a hands-on learning/practice repo (90-day ML curriculum, day-numbered notebooks). It's a sibling of `~/Desktop/Learnings`, my personal knowledge base.

## How this repo is meant to be used

- **Code and experiments live here** — notebooks, `Docs/*.md` day-logs, `Summary.md`. This repo's own git history is the record of *what was built*.
- **Conceptual learnings land in `~/Desktop/Learnings`**, not here — see that repo's `CLAUDE.md`, rule "sibling learning/practice repos". If a session is working in this repo and a genuine concept/gotcha comes up (not just "what happened today"), it should still get persisted as a `.md` note in the matching `Learnings/<Topic>/` folder, in addition to whatever gets logged here in `Docs/`.
- **Comment the code.** These notebooks are read again months later to remember what was done and why — leave technical comments explaining non-obvious choices (parameter tradeoffs, why a fix was needed, gotchas hit), not just clean identifier names.
- **Ask before committing/pushing.** Unlike the Learnings repo (which has a standing auto-commit rule), this repo doesn't — confirm before `git commit`/`git push` here.

## Environment

- `env/` — venv with GPU-enabled TensorFlow (`tensorflow[and-cuda]`). Not committed (gitignored).
- Jupyter kernel: `90daysof_ml (GPU)` — has `LD_LIBRARY_PATH` baked into its `kernel.json` so CUDA libs resolve regardless of shell state. See `~/Desktop/Learnings/Machine Learning/Local GPU Setup For TensorFlow.md` for the full gotcha writeup.
- `datasets/`, `models/`, `loaded_datasets/` — gitignored, populated locally via `kaggle datasets download`. Kaggle auth token lives at `~/.kaggle/access_token` (outside this repo).
