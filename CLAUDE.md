# CAP — Engineering principles

## Quality is non-negotiable

This pipeline produces the experimental stimuli and statistical evidence for papers targeting A* IS journals (ISR GenAI Special Issue 2026-09-07, then MISQ / JMIS / JAIS).

**Never compromise quality to save time, cost, or compute.** When a workaround would degrade output fidelity vs. fixing the root cause, fix the root cause.

Concretely, this means:

1. **Structural conditioning must be pose-keypoint based** (`controlnet_mode="pose"`), not canny / depth / identity. Canny encodes hairlines, beards, glasses, and clothing edges — these should *vary* across demographic counterfactuals; locking them creates a confound. Pin `mediapipe` and `controlnet_aux` to compatible versions in `pyproject.toml` rather than swapping to a worse preprocessor.

2. **Identity preservation must use PuLID when available.** IP-Adapter-FaceID is a fallback only. If PuLID install fails, fix the install — don't accept the fallback as default.

3. **Quantization must preserve identity fidelity.** FP8 (`bitsandbytes_8bit`) is the floor; NF4 only when L4 memory genuinely won't fit. Both wrapped in `PipelineQuantizationConfig` per the diffusers >=0.32 API.

4. **Resolution must be high enough to resolve facial features.** Floor is **768×768**. The auditees (AWS, Azure, Face++, Google Vision, DeepFace, InsightFace) all internally resize to 112–224 px for recognition and a few hundred for detection — pushing above 768² doesn't change what they consume. 512 or 224 would be too coarse to keep skin-tone, age, and identity cues sharp; 768² is the practical sweet spot on a 24 GB L4 with the full Flux + ControlNet + PuLID + IP-Adapter stack. Going higher (1024+) is fine if VRAM allows, but not required.

5. **Smoke tests are explicit and short-lived.** A smoke notebook may take quality shortcuts to validate plumbing — it must be labeled as such and never used as the production code path. MVP and full runs always use the full-quality settings.

6. **Reproducibility is a quality dimension.** Every run logs git hash, resolved config, seeds, and model version hashes (already wired in `RunManifest`). Don't add code paths that bypass the manifest.

## Priority ordering

When two of these conflict, resolve in this order:

```
quality > reproducibility > cost > speed > convenience
```

If a fix would degrade quality but save several hours, it's still the wrong fix. Surface the trade-off and pick the proper path.

## Data storage & publication strategy

Decided April 2026, ahead of MVP. The artifacts of a CAP run are split across three storage tiers, each chosen for its specific role.

| Tier | Where | Used for | Visibility |
|---|---|---|---|
| **Working storage** | Databricks Workspace Volume (`/Volumes/<catalog>/<schema>/cap/`) | Live HF model cache, in-progress runs, generated images, audit predictions during research | Private, account-scoped |
| **Publication artifacts** | HuggingFace Datasets (`alenjani/cap-counterfactuals` or similar) | Generated images + audit predictions + manifest, packaged per run | Private during research → flip to public on paper acceptance |
| **DOI / citation anchor** | Zenodo (mirrored from HF Dataset on release tag) | Citable DOI for the dataset version that backs each paper | Restricted-access (embargoed) at submission → open access on acceptance |

**Why this layout:**

- HF Datasets gives reviewers and downstream researchers a one-line `load_dataset(...)` interface; GCS doesn't.
- HF Dataset URL stays the same when flipping private → public, so the URL in a submitted manuscript stays valid post-acceptance.
- Zenodo's embargoed-DOI flow is the canonical IS-paper pattern: reviewers see the DOI exists and is registered, full data unlocks when the paper does.
- A Workspace Volume avoids a separate GCS bill and keeps the Databricks dependency footprint minimal during research. Volume → HF push is one `huggingface_hub.upload_folder(..., private=True)` call.

**Do NOT:**
- Use plain GCS for published artifacts — no DOI, no built-in viewer, costs money over project lifetime.
- Use HF Spaces for the dataset — Spaces is for demos, not data.
- Make the dataset public before paper acceptance — once public, you can't recall it for blind review.
- Skip Zenodo — most IS journals (ISR, MISQ, JMIS, JAIS) want a DOI in the data-availability statement.

**Concrete migration steps (when MVP outputs exist, not before):**

1. Create a Workspace Volume — `cap` catalog, `runs` schema, `artifacts` volume.
2. Update `configs/mvp.yaml` paths from `gs://...` → `/Volumes/cap/runs/artifacts/`.
3. Move HF cache there too (`HF_HOME=/Volumes/cap/runs/artifacts/hf_cache/`) so Flux's 24 GB only downloads once across all clusters and survives cluster rebuilds.
4. After MVP run completes: package the run dir as an HF Dataset, push as private.
5. At submission time: tag a Zenodo release of the dataset with embargoed access until the journal decision date.
6. On acceptance: flip both to public.

## Where this came from

- The quality-first principles were encoded after the first Databricks cluster smoke (April 2026), when an attempt to use `controlnet_mode="canny"` as a workaround for a `mediapipe` import bug was caught and corrected: the canny preprocessor is fundamentally unsuitable for face counterfactuals because the edges it locks (hair, beards, glasses) are demographic-correlated.
- The data storage / publication tiering was decided in the same session, before MVP, to avoid burning cycles on the wrong storage backend.
