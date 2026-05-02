# Contributing — repo conventions

Conventions for keeping plans, reviews, configs, and code traceable across the
multi-week / multi-paper lifetime of CAP. New contributors (humans or agents)
should read this once and follow it.

## Where things go

| What you're writing | Goes in | Naming |
|---|---|---|
| External or internal review of the pipeline | `reviews/` | `NNN-YYYY-MM-DD-topic.md` |
| A response to a specific review | `reviews/` (next to its review) | `NNNa-YYYY-MM-DD-topic.md` (sub-letter `a`/`b`/...) |
| A plan or strategy that supersedes a prior one | `plans/` | `NNN-topic.md` (date in front matter) |
| A running diff of "original plan vs current" | `plans/DEVIATIONS.md` (existing — append rows) | n/a |
| Master "what's running now / phase tracker" | `PROGRESS.md` (root) | n/a — there is one and only one |
| Config snapshot before a substantive rewrite | `configs/archive/` | `<name>_v<n>_<date>-archived.yaml` |
| Notebook (Databricks-side) | `notebooks/` | `NN_databricks_<purpose>.ipynb` |
| Notebook (local-side / non-Databricks) | `notebooks/` | `NN_local_<purpose>.ipynb` |
| Generator / auditor / validator code | `src/cap/<component>/` | `snake_case.py` |
| Per-task Databricks job entrypoint | `task_definitions/` | `<name>_task.py` |

## Numbering rules

- Use the **next available integer**. Don't skip numbers; don't rewrite history.
- Sub-letters `a/b/c` are for **direct follow-ups** to the same parent (e.g., `001a` is a follow-up to `001`). Don't use sub-letters for unrelated work.
- Date format is always `YYYY-MM-DD`.

## When you make a deviation from a plan

1. **Update `plans/DEVIATIONS.md`** with a row: knob, original value, new value, reason, trade-off, reversibility.
2. **If the deviation is large enough that it conceptually supersedes a prior plan**, author a new `plans/NNN-...md` and reference it from `PROGRESS.md`.
3. **Update `PROGRESS.md` phase tracker** if the deviation moves a phase forward / blocks one.

## When you change a config

1. **Snapshot the old version** to `configs/archive/<name>_v<n>_<date>-archived.yaml` *before* editing in place.
2. Edit the canonical file (`configs/<name>.yaml`).
3. Note the change in the relevant `plans/NNN-...md` and `plans/DEVIATIONS.md`.

## When you commit

- Title in present tense, imperative mood, ≤ 70 chars.
- Body explains *why*, not *what* (the diff shows what).
- If the commit relates to a specific plan / review / deviation, reference it: `(see plans/002 §3)`.

## Anti-patterns

- **Don't** put plan/review/decision docs at the repo root. Use `plans/` or `reviews/`.
- **Don't** rewrite an old plan in place — author a new numbered one. Old plans are evidence of what we considered and rejected.
- **Don't** delete configs/archive/ files. They're our trail.
- **Don't** create more than one master tracker. `PROGRESS.md` is canonical.

## Hooks / CI

The hooks in `.claude/hooks/` (if present) encode team-wide engineering standards
inherited from `.claude/CLAUDE.md`. Don't bypass them. If a hook blocks a commit,
fix the code to be compliant rather than disabling the hook.
