# Review 001 §4 — follow-up status

Tracker for the four remaining items the reviewer flagged. Updated by the
autonomous run on 2026-05-01 after the FluxPuLIDNativeGenerator architectural
fix was validated (diagnostic run 13296386370223; preserved cosine 0.63–0.75).

## Configuration deviations from the prior MVP

Documenting where the production config now differs from the pre-Review-001
MVP, so the reviewer / future-you can see what changed and why:

| Knob | Pre-review | Post-review | Why |
|---|---|---|---|
| Generator | `FluxPuLIDControlNetGenerator` (diffusers + PuLID hooks attached) | `FluxPuLIDNativeGenerator` (PuLID's native Flux + monkey-patched forward) | The diffusers transformer doesn't honor `pulid_ca` attributes — root cause of `preserved_fraction = 0`. The native Flux class does. |
| Quantization | NF4 (4-bit) | FP8 (8-bit, optimum-quanto) | CLAUDE.md item 3 prefers FP8 (closer to BF16 quality); NF4 was a workaround for an OOM that no longer occurs with the low-peak loader. |
| Resolution | 1024² | **768²** | FP8 Flux + FP8 ControlNet + PuLID adapter at 1024² peaks ~22.5 GB on a 22-GB usable L4 → would OOM. CLAUDE.md item 4 floor is 768². Auditors (DeepFace, InsightFace, AWS, etc.) internally resize to 112–224 px, so the comparison-side input is unchanged. |
| Auditors | DeepFace only | DeepFace + InsightFace | Cross-system H2 needs ≥ 2 auditors. |
| Driver host | g2-standard-16 (64 GB RAM) | g2-standard-32 (128 GB RAM) | FP8 Flux + ControlNet load peaked > 27 GB on Python heap; the smaller driver couldn't fit it past JVM/Spark overhead. Same L4 GPU. |
| Marlin FP8 kernel | enabled by default | **disabled** via monkey-patch | Marlin's per-layer packing peaks ~3× weight on GPU; cumulative load OOM'd the L4. Plain FP8 matmul ~25-30% slower; net effect: ~9 hr MVP wall time instead of ~7 hr. |

**Net direction:** generator quality went up (PuLID actually fires), quantization went up (FP8 > NF4), auditor coverage went up (2 vs 1), resolution went down (1024 → 768). The resolution change is the only quality-direction regression. Auditees consume ≤ 224 px regardless, so it's not load-bearing for the statistical claims; the only impact is that paper-figure thumbnails will be 768×768.

**Path back to 1024² if needed:** revert quantization to NF4 — fits at 1024² with FP8 ControlNet on the L4. CLAUDE.md item 3 explicitly permits NF4 when L4 doesn't fit FP8, which is now the case at 1024². Alternatively, use a bigger GPU (A100/H100) for production figures only.

## ✅ Completed (autonomous, this session)

### 1. Identity preservation (`preserved_fraction = 0.0`)
**Resolved.** The architectural cause was diffusers' `FluxTransformer2DModel`
not honoring PuLID's `pulid_ca` attribute. New `FluxPuLIDNativeGenerator`
uses PuLID's native Flux + a monkey-patched forward that applies BOTH PuLID's
identity-injection AND ControlNet residuals at the right block boundaries.

### 2. Label normalization for H1/H3 ANOVA
**Done.** `src/cap/cli/analyze.py` now normalizes gender-task predictions to
the canonical `{male, female}` alphabet before any error-rate / ANOVA
computation. Aliases handled: DeepFace (`Man`/`Woman`), AWS (`Male`/`Female`),
Face++ (`M`/`F`). Unknown values pass through unchanged.

### 3. Multi-auditor coverage (partial)
**Two auditors enabled** — DeepFace + InsightFace, both local with no API
keys. This unblocks cross-system H2 comparison at MVP scale. Cloud auditors
(AWS Rekognition, Azure Face, Face++, Google Vision) are fully implemented in
`src/cap/auditors/` and registered in `src/cap/auditors/registry.py` — adding
them is a config-only change once API credentials exist.

## 🟡 Needs user input

### 4a. Cloud auditor API credentials
To enable AWS / Azure / Face++ / Google Vision auditors, store credentials
as Databricks secrets (analogous to the existing `cap-secrets` scope) and
add the auditor to `configs/mvp.yaml`. Each auditor's required kwargs:

| Auditor          | Required env / kwargs                                  |
|------------------|--------------------------------------------------------|
| `aws_rekognition`| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, region   |
| `azure_face`     | `endpoint`, `key`                                      |
| `face_plus_plus` | `api_key`, `api_secret`                                |
| `google_vision`  | `GOOGLE_APPLICATION_CREDENTIALS` JSON                  |

Each ~$0.001-0.002 per image at 600 images = $0.60-1.20 per auditor per run.
Cheap for MVP scale, scales to ~$10-20 per auditor at full 5000-image runs.

### 4b. License
No `LICENSE` file exists in the repo. Common choices for a paper-targeted
research artifact:

- **Code** (the pipeline itself): MIT or Apache 2.0. Apache 2.0 adds an
  explicit patent grant — recommended if any reviewers/co-authors are at
  organizations sensitive to patent claims.
- **Data** (the published HF dataset): CC-BY-4.0 (most common in IS) or
  CC-BY-NC-4.0 if you want to restrict commercial use. CC-BY-4.0 only
  requires attribution, doesn't restrict downstream use, plays well with
  Zenodo.
- **Model weights**: not relevant — Flux's license already governs the
  generated images. CAP redistributes nothing trained.

**Recommended pair: Apache 2.0 (code) + CC-BY-4.0 (dataset).**
Drop in `LICENSE` (Apache 2.0 boilerplate) when ready. Add a `LICENSE-DATA`
note in the `cap-counterfactuals` HF dataset card pointing to CC-BY-4.0.

### 4c. IRB
The user should consult their institutional review board. Two scenarios:

- **Generated images only (typical case):** the published artifacts are
  synthetic faces. No real human subjects are exposed. Most IRBs treat this
  as not-human-subjects research (NHSR) and exempt it.
- **FairFace seed images:** CAP uses ~50 FairFace identities as the
  conditioning seed. FairFace is a publicly released dataset under its own
  license; the CAP pipeline does not redistribute the seed images. If the
  paper publishes side-by-side seed→generated comparisons, the seeds inherit
  FairFace's redistribution terms — usually permissive, but check.

**Action:** before paper submission, get an IRB determination letter (NHSR
or exemption) for the paper's data usage, and cite it in the methods section.
Most institutions return this in 1-2 weeks.
