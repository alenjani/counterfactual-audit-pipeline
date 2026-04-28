# Counterfactual Audit Pipeline (CAP)

A reusable scientific instrument for **counterfactual fairness auditing of facial analysis systems** using generative AI.

This is the shared infrastructure that supports a research program targeting **A\* business venues** (ISR, MISQ, JMIS, JAIS). One pipeline → multiple papers from one experimental investment.

> **Status:** Skeleton — under active development. Targeting first results for ISR GenAI Special Issue (deadline 2026-09-07).

## What it does

Given seed identities and demographic axes, CAP:

1. **Generates** counterfactual face images (same identity, varied demographics) with Flux.1 Dev + PuLID + ControlNet.
2. **Validates** identity preservation (ArcFace), image quality (FID), and attribute control (classifier residuals).
3. **Audits** generated images through commercial APIs (AWS Rekognition, Azure Face, Face++, Google Vision) and open-source models (DeepFace, InsightFace, ArcFace) on tasks: gender, age, race, emotion, identity verification, face detection, attractiveness.
4. **Analyzes** intersectional fairness with statistical rigor: ANOVA, McNemar's, ordinal logistic regression, FDR correction.
5. **Visualizes** results with publication-quality plots (matplotlib, seaborn, plotly, Altair) ready for the paper or interactive web exhibits.

## Why this exists

Existing fairness audits (e.g. Gender Shades, 2018) rely on manually curated benchmarks — expensive, demographically limited, and *correlational* rather than *causal*. CAP enables true counterfactual audits: hold identity fixed, vary only the demographic attribute, and measure whether the auditee's prediction flips.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT                                                       │
│  Seed identities (FairFace) | Text prompts                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  src/cap/generator/   Flux.1 Dev + PuLID + ControlNet       │
│  → counterfactual face images (identity preserved)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  src/cap/validator/   ArcFace identity sim + FID + attrs    │
│  → fidelity scores per generated image                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  src/cap/auditors/    7 facial-analysis systems              │
│  → predictions + confidences across tasks                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  src/cap/analysis/    Counterfactual flip rate, ANOVA,      │
│                       McNemar's, ordinal logit, FDR          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  src/cap/viz/         Publication-quality figures            │
│                       + interactive (Plotly / Altair) HTML   │
└─────────────────────────────────────────────────────────────┘
```

## Repo layout

```
counterfactual-audit-pipeline/
├── src/cap/                  # Library
│   ├── generator/            # Flux + PuLID + ControlNet wrapper
│   ├── validator/            # ArcFace, FID, attribute classifiers
│   ├── auditors/             # API clients + OSS model wrappers
│   ├── analysis/             # Statistical tests, fairness metrics
│   ├── viz/                  # Publication-quality + interactive viz
│   └── utils/                # Config loading, logging, GCS I/O, seeding
├── configs/                  # YAML run configs
│   ├── mvp.yaml              # 50 identities × 12 variations, 1 system
│   ├── full.yaml             # 200 × 60, 7 systems × 7 tasks
│   └── gcp.yaml              # GCP-specific paths + bucket
├── scripts/                  # CLI entry points
│   ├── generate.py
│   ├── audit.py
│   ├── analyze.py
│   └── visualize.py
├── docker/                   # CUDA + PyTorch image for L4
├── gcp/                      # GCE VM startup scripts
├── tests/                    # Smoke tests
├── notebooks/                # Exploratory + paper-specific analysis
├── pyproject.toml
└── README.md
```

## Running locally

```bash
# Install
pip install -e ".[dev]"

# Smoke test (no GPU needed)
pytest tests/ -v

# MVP run (requires L4 GPU)
python scripts/generate.py --config configs/mvp.yaml
python scripts/audit.py    --config configs/mvp.yaml
python scripts/analyze.py  --config configs/mvp.yaml
python scripts/visualize.py --config configs/mvp.yaml
```

## Running on GCP (L4)

```bash
# One-time: build + push image
./gcp/build_and_push.sh

# Spin up VM, run pipeline, store outputs in GCS, auto-stop
./gcp/run_on_l4.sh configs/mvp.yaml
```

VM: `g2-standard-8` (1× L4, 8 vCPU, 32 GB RAM). ~$0.70/hr. MVP run ≈ 5–8 hours ≈ ~$5.

## Reproducibility

Every run logs:
- `git commit` hash of this repo
- Resolved config snapshot (post-overrides)
- Random seeds (numpy, torch, diffusers)
- Model version hashes (Flux, PuLID, ControlNet checkpoints)
- Auditee API versions where available
- Full input → output manifest

## Data storage & publication

CAP run outputs sit in three tiers, each with a distinct role:

| Tier | Where | Visibility |
|---|---|---|
| **Working storage** | Databricks Workspace Volume (`/Volumes/cap/runs/artifacts/`) — also hosts the HF model cache so Flux's 24 GB download survives cluster rebuilds | Private |
| **Publication artifacts** | HuggingFace Datasets (e.g. `alenjani/cap-counterfactuals`) — generated images, audit predictions, manifest, packaged per run | **Private** during research → flip to **public** on paper acceptance (URL stable across the flip) |
| **DOI / citation anchor** | Zenodo (mirrored from the HF Dataset on a release tag) | **Restricted-access** (embargoed DOI) at submission → **open access** on paper acceptance |

This layout matches the canonical IS-paper workflow: reviewers see a registered DOI from submission onwards (Zenodo's embargoed-access feature), full data unlocks when the journal decision does, and downstream researchers get a one-line `load_dataset("alenjani/cap-counterfactuals")` interface.

```python
# Once paper is accepted and dataset is public:
from datasets import load_dataset
ds = load_dataset("alenjani/cap-counterfactuals", split="mvp")
```

> The earlier `configs/gcp.yaml` / `gcp/` paths predate this decision. They still work for direct GCE runs but **published artifacts go to HF + Zenodo, not GCS.** See `CLAUDE.md` for the full rationale.

## Papers using this pipeline

- **Paper 1** — *Counterfactual Fairness Auditing of Facial Analysis Systems Using Generative AI* (target: ISR GenAI Special Issue, 2026-09-07)
- **Paper 2** — Emotion recognition bias as IS governance (planned)
- **Paper 3** — Commercial vs open-source IT sourcing under fairness constraints (planned)
- **Paper 4** — Longitudinal re-audit, T+18 months (planned)

## License

TBD — leaning toward MIT or Apache 2.0 to maximize reuse.

## Citation

> Alenjani, A. (2026). *Counterfactual Fairness Auditing of Facial Analysis Systems Using Generative AI: A Design Science Approach.* Manuscript under review at Information Systems Research.
