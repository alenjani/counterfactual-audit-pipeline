# PROGRESS

Master tracker for CAP. Skim top-to-bottom to know where things stand.

## What's running NOW

| status | item | run_id | notes |
|---|---|---|---|
| 🔄 RUNNING | MVP at paper-grade settings (600 images) | 96570565797190 | ~9 hr ETA from launch 2026-05-02 12:34. On success → notebook 07 auto-chains. |

## Where to find things

```
plans/
  001-original-research-plan.md          ← what was originally planned
  002-paper-grade-architectural-fix.md   ← May 2026 architectural pivot
  003-full-instrument-plan.md            ← 36K full-run execution plan
  DEVIATIONS.md                          ← original ↔ current diff with rationale
reviews/
  001-2026-04-30-mvp-review.md           ← original review
  001a-2026-05-02-followups.md           ← response to review 001
configs/
  mvp.yaml                               ← current MVP config (paper-grade)
  full.yaml                              ← full-run config (TO BE UPDATED post-MVP)
  archive/full_v1_2026-04-archived.yaml  ← original full-run config preserved
src/cap/                                 ← Python package
notebooks/                               ← Databricks notebooks (smoke, MVP, diagnostic, downstream)
results/mvp_demo/                        ← First MVP outputs (April 2026, broken backend — KEEP for posterity)
```

## Project phase tracker

| phase | status | done date | next |
|---|---|---|---|
| 0. Project scaffold | ✅ done | ~2026-04 | — |
| 1. First MVP run (broken backend) | ✅ done | 2026-04-30 | — |
| 2. Review 001 | ✅ done | 2026-04-30 | — |
| 3. Architectural fix (FluxPuLIDNativeGenerator) | ✅ done | 2026-05-01 | — |
| 4. Diagnostic validation (n=2) | ✅ done | 2026-05-01 | — |
| 5. MVP-6 (n=600, paper-grade fix) | ✅ done | 2026-05-02 | — |
| 6. Smoke validation of analysis pipeline (n=36) | ✅ done | 2026-05-02 | — |
| 7. **MVP at paper-grade settings (current)** | 🔄 RUNNING | (in progress) | analysis verdict + decide if quality is locked |
| 8. `configs/full.yaml` revision | ⏳ pending phase 7 | — | author once paper-grade settings confirmed at MVP scale |
| 9. Cloud auditor API credentials | ⏳ user action | — | AWS, Azure, Google, Face++ keys → `cap-secrets` |
| 10A. Full gen — Stage A (200 IDs × 12 axes × seed 42 = 2,400) | ⏳ pending 7-9 | — | ~1.5 days |
| 10A-bk. HF backup → `cap-counterfactuals-stage-a` | ⏳ | — | ~5 min |
| 10B. Full gen — Stage B (+9,600 with age axis × seed 42 = 12,000 cumulative) | ⏳ | — | +5.5 days |
| 10B-bk. HF backup → `cap-counterfactuals-stage-b` | ⏳ | — | ~25 min |
| 10C. Full gen — Stage C (+24,000 with seeds 137 + 2718 = 36,000 cumulative) | ⏳ | — | +14 days |
| 10C-bk. HF backup → `cap-counterfactuals-stage-c` | ⏳ | — | ~75 min |
| 11. Full audit (7 auditors × 36K images) | ⏳ | — | ~24 hr (bottlenecked by Google Vision rate limit) |
| 12. Full analysis | ⏳ | — | <1 hr |
| 13. ISR paper drafting (Sept 7 deadline) | ⏳ | — | uses `paper_1_isr` slice |
| 14. License + IRB | ⏳ user action | — | non-blocking; needed before publication |

## Outstanding user decisions (from `reviews/001a-2026-05-02-followups.md`)

- [ ] Cloud auditor API credentials → store in Databricks `cap-secrets`
- [ ] License: Apache 2.0 (code) + CC-BY-4.0 (data) recommended
- [ ] IRB determination letter from institution

## Cost / budget tracker

| line item | budgeted | spent (est.) | notes |
|---|---|---|---|
| Cluster (4× L4 + driver) | $2,500 | TBD | 525 GPU-hr × ~$5/hr |
| Cloud auditor APIs | $300 | $0 | not yet running |
| HF / Zenodo | $0 | $0 | free tiers |
| **Total** | **$3,000** | TBD | |

## Quick links

- [Cluster page](https://2546847502462311.1.gcp.databricks.com/compute/clusters/6106-192556-lj0uddmy?o=2546847502462311)
- [HF dev dataset](https://huggingface.co/datasets/alenjani/cap-counterfactuals-dev) (private)
- [GitHub repo](https://github.com/alenjani/counterfactual-audit-pipeline)
- ISR Special Issue deadline: 2026-09-07

## How to use this file

- Update the "What's running NOW" row when starting/finishing major runs.
- Move phases between `⏳ pending` → `🔄 in progress` → `✅ done` as they complete.
- Whenever you find yourself making a deviation from `plans/001`, log it in `plans/DEVIATIONS.md` so the original-vs-current trace stays clean.
- When a plan changes substantially (vs minor knob tweaks), author a new `plans/NNN-...md` and link it here.
