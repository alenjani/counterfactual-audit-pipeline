# 002 — Paper-grade architectural fix (May 2026)

**Date authored:** 2026-05-02
**Supersedes parts of:** `plans/001-original-research-plan.md`
**Companion:** `plans/003-full-instrument-plan.md` (execution); `reviews/001a-2026-05-02-followups.md` (review response)
**Status:** ACTIVE — informs all future MVP and full-run executions

---

## Why this plan exists

The original plan (`plans/001-original-research-plan.md`) called for `flux_pulid_controlnet` as the generator backend. The April 2026 MVP run produced `preserved_fraction = 0.0` — i.e., generated counterfactuals don't preserve the seed identity at all, making any downstream audit measurement meaningless. Review 001 §4 flagged this as the most critical issue blocking publication.

Investigation showed the cause: PuLID's identity-injection hooks (`pulid_ca`) attach as attributes to the diffusers `FluxTransformer2DModel`, but **the diffusers transformer's `forward()` never reads those attributes**. PuLID's *own* native `flux.model.Flux` does. So the PuLID-on-diffusers integration was a no-op; generation ran as plain text-to-image with no identity injection.

This document captures the architectural changes that fixed it, plus the cascade of hardware-forced configuration changes that followed.

## Architectural fix

### `FluxPuLIDNativeGenerator` (new)

- File: `src/cap/generator/flux_pulid_native.py`
- Uses PuLID's native `flux.model.Flux` (which honors `pulid_ca`)
- Adds a monkey-patched `forward()` that *also* applies diffusers-format ControlNet residuals at the right block boundaries
- Result: PuLID identity injection AND ControlNet pose conditioning both work in a single forward pass

### Validation

- Diagnostic notebook (`06_databricks_pulid_diagnostic`, run 13296386370223, 2026-05-01):
  - Two seeds × two generations
  - ArcFace cosine: seed A vs image A = **0.7525**, seed B vs image B = **0.6277**
  - Image A vs image B cosine = -0.0019 (identities clearly distinct)
- MVP run-6 at 600 images: preserved_fraction = **0.807** at cosine ≥ 0.5; mean cosine 0.578.
- Smoke run at 36 images with paper-grade settings: mean cosine **0.658**, preserved_fraction = **1.00** at 0.5; 0.778 at 0.6.

## Hardware-forced configuration changes

The 22 GB-usable L4 (driver and worker GPUs) imposes these constraints; each finding cost a debug cycle to discover:

| Constraint | Forced setting | Trade |
|---|---|---|
| FP8 Flux load OOMs CPU at 36 GB peak (BF16 alloc + FP8 sd) | Custom low-peak loader: `meta` device + direct-to-GPU FP8 via safetensors | None — peak drops to 12 GB |
| `optimum-quanto.requantize` materializes BF16 on GPU before loading FP8 (24 → 22 GB OOM) | Skip the BF16 intermediate via custom `_load_flux_quintized_low_peak` | None — fits 12 GB FP8 directly |
| Marlin FP8 packing peaks ~3× weight per layer on GPU | Monkey-patch `WeightQBytesTensor.optimize` AND `.create` to skip Marlin | Plain FP8 matmul ~25-30% slower |
| Marlin JIT compile fails on read-only worker `ephemeral_nfs` | Patch above also avoids the JIT entirely | None |
| 1024² + FP8 Flux + FP8 ControlNet + PuLID = ~22.5 GB → OOM | Drop to **768² resolution** | CLAUDE.md item 4 floor met (auditors resize to ≤224 px anyway); paper figures will be smaller |
| `/local_disk0` (~79 GB) fills with HF cache + PuLID weights | Redirect torch JIT extensions + TMPDIR to `/dev/shm` (32 GB tmpfs) | ~1-2 GB RAM consumed |
| GCS-FUSE Volume rejects HF Hub `.incomplete` rename | Force `HF_HOME=/local_disk0/hf_cache` per Ray actor | Cold per-worker re-download on each cluster restart (~5 min × 12 GB) |
| facexlib + PuLID write to package weights/ dir which is read-only on workers | Monkey-patch `FaceRestoreHelper.__init__` + `init_parsing_model` to redirect `model_rootpath` to `/local_disk0` | None |

## Quality settings (validated paper-grade)

| Knob | Old plan | New plan | Why |
|---|---|---|---|
| `generator.backend` | `flux_pulid_controlnet` | `flux_pulid_native` | Old broken; new validated. |
| `dtype` | `fp8` | `fp8` | same |
| `num_inference_steps` | 28 | **40** | Lifts mean cosine 0.58 → 0.66 in smoke. Closer to "paper-grade" thresholds (0.65+). |
| `id_weight` | (default 1.0) | **1.5** | Stronger PuLID injection. |
| `controlnet_conditioning_scale` | (default 0.6) | 0.6 (explicit) | unchanged value, made explicit |
| `width × height` | 1024 × 1024 | **768 × 768** | Hardware-forced; CLAUDE.md item 4 floor met. |
| Marlin FP8 kernel | Implied on | **Disabled** in code | Hardware-forced. |
| Driver host | (default = same as worker) | **`g2-standard-32`** (128 GB) | Original `g2-standard-16` (64 GB) too small for FP8 load on driver. Same L4 GPU. |

## DeepFace + Keras 3 fix

DeepFace's age/gender/emotion models break with Keras 3 (default in TF 2.16+) — `KerasTensor cannot be used as input to a TensorFlow function`. Fix in code:

- `src/cap/auditors/deepface_local.py`: set `TF_USE_LEGACY_KERAS=1` + `CUDA_VISIBLE_DEVICES=-1` before TF imports; call `tf.config.set_visible_devices([], "GPU")` defensively.
- TF runs on CPU only (audit is fast enough on CPU; avoids GPU contention with PyTorch).

## Label normalization for H1/H3 ANOVA

Per Review 001 §4 follow-up: DeepFace returns `Man`/`Woman`, `axis_gender` stores `male`/`female`. Without normalization, every prediction marked as error → zero variance → empty H1/H3.

Fix in `src/cap/cli/analyze.py`: explicit `_GENDER_ALIASES` map normalizing `man → male`, `woman → female`, `m → male`, etc., applied before any error/ANOVA computation.

## Wall-time consequence

Original config commented "150–300 GPU-hours" assuming 28 steps + 1024² + Marlin.

Actual settings forced by hardware: 40 steps + 768² + no-Marlin → **~525 GPU-hours for the full 36K instrument** on the current 4× L4 cluster. ~2-3× longer than the original estimate.

Path-to-A100 (future cost optimization): Marlin works + 1024² + 28 steps would all fit. Wall drops back to ~125-150 GPU-hours. Out of scope until needed.

## Cost estimate

| line item | budget |
|---|---|
| Cluster: 525 GPU-hr × ~$5/hr (4× L4 + driver) | ~$2,500 |
| Cloud auditor APIs (AWS, Azure, Google Vision, Face++): 36K images × 4 auditors | ~$140-300 |
| HuggingFace Datasets + Zenodo | $0 (free tiers sufficient) |
| **Total** | **~$2,800-3,000** |

## Open items (need user input)

- Cloud auditor API credentials → Databricks `cap-secrets` scope
- License choice (recommend: Apache 2.0 for code, CC-BY-4.0 for data)
- IRB determination letter (institutional, ~1-2 wk turnaround)

These don't block compute; can run in parallel with the multi-week generation.
