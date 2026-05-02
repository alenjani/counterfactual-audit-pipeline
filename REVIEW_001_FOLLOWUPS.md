# Review 001 §4 — follow-up status

Tracker for the four remaining items the reviewer flagged. Updated by the
autonomous run on 2026-05-01 after the FluxPuLIDNativeGenerator architectural
fix was validated (diagnostic run 13296386370223; preserved cosine 0.63–0.75).

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
