# CAP — Engineering principles

## Quality is non-negotiable

This pipeline produces the experimental stimuli and statistical evidence for papers targeting A* IS journals (ISR GenAI Special Issue 2026-09-07, then MISQ / JMIS / JAIS).

**Never compromise quality to save time, cost, or compute.** When a workaround would degrade output fidelity vs. fixing the root cause, fix the root cause.

Concretely, this means:

1. **Structural conditioning must be pose-keypoint based** (`controlnet_mode="pose"`), not canny / depth / identity. Canny encodes hairlines, beards, glasses, and clothing edges — these should *vary* across demographic counterfactuals; locking them creates a confound. Pin `mediapipe` and `controlnet_aux` to compatible versions in `pyproject.toml` rather than swapping to a worse preprocessor.

2. **Identity preservation must use PuLID when available.** IP-Adapter-FaceID is a fallback only. If PuLID install fails, fix the install — don't accept the fallback as default.

3. **Quantization must preserve identity fidelity.** FP8 (`bitsandbytes_8bit`) is the floor; NF4 only when L4 memory genuinely won't fit. Both wrapped in `PipelineQuantizationConfig` per the diffusers >=0.32 API.

4. **Resolution must match the audited systems.** Default 1024×1024 — don't drop to 512 or 768 to save VRAM. If memory is tight, prefer NF4 or a different GPU over reducing resolution.

5. **Smoke tests are explicit and short-lived.** A smoke notebook may take quality shortcuts to validate plumbing — it must be labeled as such and never used as the production code path. MVP and full runs always use the full-quality settings.

6. **Reproducibility is a quality dimension.** Every run logs git hash, resolved config, seeds, and model version hashes (already wired in `RunManifest`). Don't add code paths that bypass the manifest.

## Priority ordering

When two of these conflict, resolve in this order:

```
quality > reproducibility > cost > speed > convenience
```

If a fix would degrade quality but save several hours, it's still the wrong fix. Surface the trade-off and pick the proper path.

## Where this came from

Encoded after the first Databricks cluster smoke (April 2026), when an attempt to use `controlnet_mode="canny"` as a workaround for a `mediapipe` import bug was caught and corrected: the canny preprocessor is fundamentally unsuitable for face counterfactuals because the edges it locks (hair, beards, glasses) are demographic-correlated.
