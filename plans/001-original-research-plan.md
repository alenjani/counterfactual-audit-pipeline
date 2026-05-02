# 001 — Original research plan

**Date authored:** ~2026-04 (project inception)
**Source artifacts:** `configs/archive/full_v1_2026-04-archived.yaml`, `reviews/001-2026-04-30-mvp-review.md`
**Status:** SUPERSEDED — see `plans/002-paper-grade-architectural-fix.md`, `plans/003-full-instrument-plan.md`

---

## Goal

Build CAP (Counterfactual Audit Pipeline) as a **reusable scientific instrument across four planned papers**. The core idea: stratify a balanced set of seed identities, generate counterfactual variants along demographic axes (skin tone × gender × age), audit the variants with multiple commercial + open-source face APIs, and statistically test whether auditor errors depend on the demographic axis.

## Quantitative scope

- **Generation**: 200 seed identities × 60 axis combos (6 skin × 2 gender × 5 age) × 3 generation seeds = **18,000 images** (the `≈18K` comment in old config was off — actual product is 36,000).
- **Audit**: 7 auditors × 7 tasks ≈ **882K to 1.76M predictions** depending on per-task call multiplicity.
- **Analysis**: bootstrap CI (5000 iterations), per-paper slices.

## Per-paper slicing (planned at instrument time)

| paper | auditor subset | tasks | axes |
|---|---|---|---|
| **Paper 1 (ISR)** — methodology + bias audit | aws_rekognition, azure_face, deepface, insightface | gender, age | skin_tone × gender |
| **Paper 2** — emotion bias | all 7 | emotion | skin_tone × gender × age |
| **Paper 3** — OSS vs commercial auditor disagreement | all 7 | gender, age, race | grouped by commercial_vs_oss |
| **Paper 4** — TBD | TBD | TBD | TBD |

## Generation pipeline (as planned)

- **Backend**: `flux_pulid_controlnet` — diffusers `FluxControlNetPipeline` + PuLID hooks attached
- **Quantization**: `dtype: fp8` via diffusers `PipelineQuantizationConfig`
- **Resolution**: 1024 × 1024
- **Inference steps**: 28
- **Guidance scale**: 3.5
- **Hardware**: 4× L4 (1 driver + 3 workers, Ray fan-out)
- **Expected wall time**: ~150–300 GPU-hours (per the old config comment)

## Storage & publication strategy

- Working: Databricks Workspace Volume `/Volumes/ds_work/alenj00/cap_cache/`
- Publication: HuggingFace Datasets (private → public on paper acceptance)
- DOI: Zenodo (embargoed → open on paper acceptance)

## What changed vs this plan

See `plans/DEVIATIONS.md` for the running comparison.

Headline changes documented in `plans/002-paper-grade-architectural-fix.md`:
- `flux_pulid_controlnet` was **architecturally broken** (Review 001 §4: PuLID hooks don't fire on diffusers' transformer). Replaced with `flux_pulid_native`.
- 1024² + Marlin FP8 doesn't fit on L4 → forced 768² + no-Marlin.
- 28 steps wasn't sufficient for paper-grade preserved fraction → bumped to 40 steps.
- Net wall-time impact: ~150 GPU-hours estimate → ~525 GPU-hours actual.
