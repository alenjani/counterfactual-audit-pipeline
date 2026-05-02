# 003 — Full instrument execution plan

**Date authored:** 2026-05-02
**Companion:** `plans/002-paper-grade-architectural-fix.md` (architecture); `plans/001-original-research-plan.md` (original intent)
**Status:** ACTIVE — kicked off after MVP validates paper-grade quality

---

## Decision: one big run, gradual publication

**Generation is monolithic.** All 36,000 images get generated in a single (multi-week) production run on the existing cluster, audited end-to-end with all 7 auditors, and analyzed comprehensively. **Per-paper publication slicing happens AFTER analysis** at submission time, not during generation. This avoids re-generating overlapping cells and keeps the dataset internally consistent.

## What gets generated

- **200 seed identities** stratified across FairFace by race × gender × age
- **60 axis combinations per identity**: 6 skin × 2 gender × 5 age
- **3 generation seeds per cell** for noise robustness
- **= 36,000 images total**

## What gets audited

- **7 auditors**: aws_rekognition, azure_face, google_vision, face_plus_plus, deepface, insightface, arcface
- **7 tasks**: gender, age, race, emotion, identity_verification, face_detection, attractiveness
- **= ~1.76M predictions** (some auditor/task combinations don't apply; effective count slightly lower)

## What gets analyzed (per analyze.py)

- Coarse and pairwise flip rates per (auditor, task)
- H1: two-way ANOVA on error rate × (skin_tone × axis_gender)
- H2: McNemar's paired test for cross-system disagreement
- H3: ordinal logit regression for skin-tone monotonicity
- FDR correction across the full family of tests
- Per-paper slices (paper_1_isr, paper_2_emotion, paper_3_oss_vs_commercial) computed and exported

## Execution timeline

### Wall time

- Generation: 36,000 images × 3.5 min/image / 4 actors = ~525 GPU-hours = **~22 days continuous compute**
- Audit (cloud): bottlenecked by Google Vision rate limit, ~24 hr total
- Audit (local): parallelizable on cluster, ~8-10 hr
- Analyze + viz: <1 hr

**Critical path: generation ≈ 22 days. Everything else is small in comparison.**

### Three-stage cumulative generation (locked 2026-05-02)

The 36K is built incrementally — never re-doing prior work — so an interruption at any point leaves a usable dataset:

| stage | new cells | new images | cumulative images | wall (delta) | cumulative wall | papers usable if interrupted here |
|---|---|---|---|---|---|---|
| **A** | 200 IDs × 12 (skin × gender) × seed 42 | 2,400 | 2,400 | ~1.5 d | 1.5 d | Paper 1 ISR (single-seed) |
| **B** | 200 IDs × 48 (= 60 - 12) NEW axes × seed 42 | 9,600 | 12,000 | ~5.5 d | 7 d | Papers 2 + 3 + Paper 1 with age axis (single-seed) |
| **C** | All 12,000 cells × seeds 137, 2718 | 24,000 | 36,000 | ~14 d | 21 d | All 4 papers, 3-seed robustness |

Paper 1 (ISR) only needs Stage A (single seed) to ship; reviewer-defensible robustness comes after Stage C.

### HF Datasets backup checkpoints

After each stage completes, push a **versioned snapshot** to HuggingFace Datasets (private). This is off-cluster backup against Volume corruption / accidental deletion (the latter happened to MVP-6 artifacts; cause unexplained).

| after stage | HF dataset push | content | est. push time |
|---|---|---|---|
| A | `alenjani/cap-counterfactuals-stage-a` (private) | 2,400 PNGs + manifests | ~5 min |
| B | `alenjani/cap-counterfactuals-stage-b` (private) | 12,000 PNGs + manifests + analysis-so-far | ~25 min |
| C | `alenjani/cap-counterfactuals-stage-c` (private) | 36,000 PNGs + full audit + analysis + viz | ~75 min |

Each stage's dataset is a separate HF repo — snapshots are immutable, so you can always roll back to a prior stage's data exactly. The final Zenodo DOI (at first paper submission) gets minted from `stage-c`.

### Calendar

- Today: 2026-05-02
- ISR deadline: 2026-09-07 (~128 days out)
- 22 days of cluster time fits comfortably with breaks

| weeks from now | activity |
|---|---|
| 0 (today) | MVP run completes (~600 images, paper-grade settings validated) |
| 1 | Update `configs/full.yaml` to FluxPuLIDNativeGenerator + paper-grade. Kick off full generation. |
| 1-5 | Full generation (~22 days continuous, split across multiple Databricks job submissions; skip-if-exists handles resume) |
| 5-6 | Run all 7 auditors |
| 6 | Full analysis + visualization |
| 7-8 | ISR paper drafting (using paper_1_isr slice) |
| 7-8 | License + IRB + dataset publication prep |
| 9 | ISR submission |

Buffer: ~5 weeks against the deadline.

## Cluster orchestration for the long generation

Databricks single-job timeout is 18 hours. A 22-day generation needs multiple sequential submissions. The generator's `skip-if-exists` logic naturally handles resumption.

**Pattern:**

```python
while pngs_present < 36000:
    submit_run(notebook="02_databricks_mvp_run", config="full.yaml")
    poll_until_terminal()
    if cluster_failed:
        diagnose_and_retry()
    pngs_present = count_pngs(volume_path)
```

A wrapper script will be authored once `configs/full.yaml` is updated.

### Failure modes to plan for

- **GCP STOCKOUT**: hit 6+ times in May 2026. Build in 30% buffer.
- **DBR runtime updates**: pin runtime version `16.4.x-gpu-ml-scala2.12` in the cluster spec.
- **Volume FUSE quirks** (already mitigated in code): no append, no rename, no long-lived handles.
- **HF Hub rate limits**: Each cluster cold-start re-downloads 12 GB FP8 to per-worker `/local_disk0`. With `HF_HUB_ENABLE_HF_TRANSFER=1`, ~5 min per worker.
- **Per-worker `/local_disk0` fills**: Schedule periodic `cap_clean` notebook runs to keep `/local_disk0` < 80% full.

## Publication staging (post-generation)

Once the full instrument is built and analyzed:

| paper | slice | est. submission |
|---|---|---|
| Paper 1 (ISR) — methodology + small empirical | `paper_1_isr` slice (4 auditors, gender/age tasks, skin × gender axes) | 2026-09-07 |
| Paper 2 — emotion bias deep dive | `paper_2_emotion` slice (all auditors, emotion task, all axes) | TBD |
| Paper 3 — OSS vs commercial auditor disagreement | `paper_3_oss_vs_commercial` slice (all auditors, gender/age/race) | TBD |
| Paper 4 — TBD | TBD | TBD |

## Storage tiering (per CLAUDE.md)

- **Working storage**: Databricks Workspace Volume `/Volumes/ds_work/alenj00/cap_cache/`
- **Publication artifacts**: HuggingFace Datasets `alenjani/cap-counterfactuals` (private during research → public on first paper acceptance)
- **DOI / citation anchor**: Zenodo (embargoed → open on acceptance)
- **Dev iteration**: HF Datasets `alenjani/cap-counterfactuals-dev` (private; staging for the research team)

## Cost budget

| line item | est. |
|---|---|
| Cluster (525 GPU-hr × ~$5/hr) | ~$2,500 |
| Cloud auditors (4 cloud × 36K × ~$0.0015) | ~$140-300 |
| Buffer (debugging, re-runs, cluster STOCKOUT idle) | ~$500 |
| **Total** | **~$3,000** |

## Outstanding decisions (needs user input)

- Whether to start the full run NOW (after MVP validates) or wait for: (a) cloud auditor API credentials, (b) license decision, (c) IRB determination. Recommendation: start generation while those proceed in parallel — generation doesn't depend on them.
- Whether to use 1024² instead of 768² IF an A100 cluster becomes available. Currently 768² for the L4. Re-evaluate if cluster upgrade happens.
