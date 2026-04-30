# MVP Demo Outputs

A snapshot of one full run of the Counterfactual Audit Pipeline at MVP scale,
with everything-but-the-images committed in-repo for instant inspection.

## Run config (from `configs/mvp.yaml`)

- 50 FairFace seed identities × 12 counterfactual variations = **600 generated images**
- Generator: Flux.1-dev + ControlNet-Union (pose) + PuLID + InsightFace antelopev2
- Quantization: NF4, resolution 1024², 28 inference steps
- Auditor: DeepFace (gender + age)
- Hardware: 4× L4 GPUs on Databricks (Ray fan-out)

## What's here vs what's not

| Tier | Where |
|---|---|
| Tiny — committed to this repo | `analysis/`, `validation/`, `audit/*.parquet`, `viz/figures/`, `viz/dashboard.html`, `generated/manifest.parquet`, **12 sample PNGs** under `generated/samples/` |
| Large — NOT in this repo | All 600 generated PNGs (~600 MB). Lives on a Databricks Workspace Volume, will move to a HuggingFace Dataset on paper submission. |

## Directory map

```
results/mvp_demo/
├── README.md                    ← this file
├── generated/
│   ├── manifest.parquet         ← full 600-image index (paths, axes, metadata)
│   └── samples/                 ← 12 PNGs from one seed identity (full demographic factorial)
├── validation/
│   ├── identity_scores.parquet  ← ArcFace cosine sim for all 600 (seed, counterfactual) pairs
│   ├── summary.json             ← aggregate identity-preservation stats
│   └── run_manifest.json
├── audit/
│   ├── predictions.parquet      ← 1200 DeepFace predictions (gender + age × 600 images)
│   └── run_manifest.json
├── analysis/
│   ├── flip_rates.csv           ← coarse 0/1 flip-rate per (auditor, task)
│   ├── pairwise_flip_rates.csv  ← effect-size: fraction of pair-comparisons that disagree
│   ├── pairwise_per_identity_*.csv
│   ├── intersectional_deepface_gender.csv
│   ├── h1_anova.csv             ← H1: skin_tone × gender ANOVA (skipped — known issue, see below)
│   ├── h2_mcnemars.csv          ← H2: McNemar's paired flip test (skin tone 1 vs 6)
│   ├── h3_ordinal_logit.csv     ← H3: continuous skin tone effect (skipped — known issue)
│   ├── summary.json             ← all-results JSON summary
│   └── run_manifest.json
└── viz/
    ├── dashboard.html           ← interactive Plotly dashboard (open in browser)
    └── figures/
        └── intersectional_deepface_gender.{pdf,svg,png}
```

## Headline result (paper-grade)

From `analysis/h2_mcnemars.csv`:

> DeepFace gender prediction flips significantly between Fitzpatrick I and Fitzpatrick VI
> on PuLID-controlled identity counterfactuals: McNemar's statistic = ∞, p = 0.0,
> p_FDR = 0.0 → **H2 supported**.

From `analysis/pairwise_flip_rates.csv`:

| auditor | task | mean_pairwise_flip_rate |
|---|---|---|
| deepface | gender | **0.540** |
| deepface | age | 0.845 |

## Known issues (do not ship as-is for the paper)

1. **Identity preservation = 0% at θ=0.5.** `validation/summary.json` shows
   `mean_cosine_similarity ≈ -0.004` (random). Means PuLID/IP-Adapter did not
   actually inject identity in this MVP run. Likely PuLID silently fell back to
   weak IP-Adapter at low scale. Must debug before the full paper run.
2. **H1 ANOVA + H3 logit empty.** Zero-variance issue: DeepFace returns
   "Man"/"Woman" but `axis_gender` stores "male"/"female" → every prediction
   marked `error=1`, no variance for ANOVA to fit. Need a label-normalization
   map per auditor (small follow-up).
3. **Single auditor.** Add AWS Rekognition / Azure Face / Face++ / Google Vision
   for the cross-system H2 comparison the paper actually wants.

## How to reproduce

```bash
# Generate (4× L4 GPUs via Databricks Ray):
cap-generate --config configs/mvp.yaml --backend ray --num-actors 4 \
  --cache-dir /Volumes/.../hf_cache --pulid-src /Volumes/.../pulid_src

# Validate identity preservation:
cap-validate --config configs/mvp.yaml --cache-dir /Volumes/.../hf_cache

# Audit:
cap-audit --config configs/mvp.yaml

# Analyze + visualize:
cap-analyze --config configs/mvp.yaml
cap-visualize --config configs/mvp.yaml
```

Or use the Databricks notebooks: `notebooks/02_databricks_mvp_run.ipynb` (generate)
and `notebooks/04_databricks_validate_analyze_viz.ipynb` (validate + analyze + viz).
