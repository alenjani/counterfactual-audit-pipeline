# Deviations log

Running comparison of original plan (`plans/001-original-research-plan.md`) vs the active execution plan (`plans/002-paper-grade-architectural-fix.md` + `plans/003-full-instrument-plan.md`). Each row tracks a specific deviation: what changed, why, what trade-off was accepted.

| # | knob | original (plan 001) | current (plan 002) | reason | trade-off accepted | reversible? |
|---|---|---|---|---|---|---|
| 1 | Generator backend | `flux_pulid_controlnet` (diffusers + PuLID hooks attached) | `flux_pulid_native` (PuLID's native Flux + monkey-patched forward) | Original architecturally broken — PuLID hooks don't fire on diffusers transformer (Review 001 §4) | Engineering complexity (custom forward, monkey-patches for Marlin/facexlib) | No — fix is required |
| 2 | Quantization | FP8 (intent) | FP8 (validated in code) | same | none | n/a |
| 3 | Inference steps | 28 | **40** | 28 gave mean cosine 0.578 (MVP-6, 80% preserved). 40 lifts to 0.66+ (smoke run, 100% preserved at 0.5 / 78% at 0.6) — closer to paper-grade thresholds | +43% wall time per image | Yes (drop back to 28 if time-pressed; expect ~80% preserved instead of 90%+) |
| 4 | Resolution | 1024 × 1024 | **768 × 768** | 1024² + FP8 Flux + FP8 ControlNet + PuLID = ~22.5 GB → OOM on 22 GB-usable L4 | Auditors resize to ≤224 px so the comparison-side input is unaffected. Paper figures will be at 768² | Yes IF GPU upgrade happens (A100 80GB fits 1024²) |
| 5 | `id_weight` (PuLID strength) | (default 1.0) | **1.5** | Stronger PuLID injection. Combined with #3, lifts preserved fraction to paper-grade | Slight reduction in ControlNet adherence (negligible per smoke run inspection) | Yes |
| 6 | Marlin FP8 kernel | implied enabled | **disabled** (in code) | Marlin's per-layer pack peaks ~3× weight on GPU → cumulative load OOMs L4. Plus JIT compile fails on read-only worker `ephemeral_nfs` | ~25-30% slower matmul. Net 1.3× wall time | Yes IF GPU upgrade happens |
| 7 | Driver host | `g2-standard-16` (64 GB RAM) | **`g2-standard-32`** (128 GB) | FP8 Flux load peaks ~36 GB on host RAM (transient BF16 + FP8 sd). 64 GB driver minus JVM/Spark = ~27 GB Python heap → OOM | ~2× driver cost | Yes (revert if GPU model changes / load profile shrinks) |
| 8 | DeepFace TF backend | (default = GPU if available) | **CPU only + Keras 2** | TF + PyTorch contend for L4 CUDA context (`FusedBatchNormV3` graph error). Plus Keras 3 incompatible with DeepFace's models | Slightly slower audit; not on critical path | Yes (revert when DeepFace + Keras 3 become compatible) |
| 9 | Wall time estimate | "~150-300 GPU-hours" (config comment) | **~525 GPU-hours** | Net effect of #3 (×1.43) + #6 (×1.3) + #4 (slightly faster from smaller activations) | ~22 days continuous vs ~6-12 days | Yes (with A100 + Marlin: drops to ~125-150 GPU-hours per original) |
| 10 | Storage paths | `outputs/full/...` (relative) | `/Volumes/ds_work/alenj00/cap_cache/runs/full/...` (Volume) | Relative paths don't resolve on Databricks Ray actors (different worker hosts) | None — Volume is the intended working storage per CLAUDE.md | n/a |

## What did NOT change vs original plan

- **Total images**: 36,000 ✓
- **Identities**: 200 ✓
- **Axes**: 6 skin × 2 gender × 5 age = 60 per identity ✓
- **Generation seeds**: 3 (42, 137, 2718) ✓
- **Auditors**: 7 (4 cloud + 3 local) ✓
- **Tasks**: 7 ✓
- **Analysis methodology**: H1 ANOVA, H2 McNemar's, H3 ordinal logit, FDR, 5000-iter bootstrap ✓
- **Storage tiering**: Volume → HF Datasets → Zenodo ✓
- **Per-paper slicing**: paper_1_isr, paper_2_emotion, paper_3_oss_vs_commercial ✓

## Net direction

- **Quality**: ✅ Higher than original — quantization choice held (FP8), identity preservation actually works now (was 0% before), and validated paper-grade preservation cosine.
- **Resolution**: 🔻 Lower than original (768 vs 1024). Within CLAUDE.md spec (item 4 floor); not on critical path for any auditor (which all resize ≤224 px).
- **Wall time**: 🔻 Slower than original estimate (525 vs 150-300 GPU-hr). Forced by L4 hardware ceiling.
- **Cost**: 🔻 Higher than implied (~$3K vs unestimated). Within reasonable research budget.

The science (image count, identities, axes, seeds, auditors, tasks, statistical tests) is unchanged. The deviations are all engineering / hardware accommodations.
