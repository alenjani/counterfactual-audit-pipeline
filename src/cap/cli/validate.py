"""CLI: validate generated counterfactuals.

For each (seed, counterfactual) pair, compute:
  - ArcFace cosine similarity (identity preservation) — required for the paper

Future extensions (FID, attribute classifier residuals) can be wired here.

Output: <validation_dir>/identity_scores.parquet  (one row per generated image,
joined back to the generation manifest).
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from cap.data import load_or_sample_seeds
from cap.utils import RunManifest, get_logger, load_config
from cap.validator import ArcFaceIdentityValidator

logger = get_logger()


@click.command()
@click.option("--config", "config_path", required=True)
@click.option("--limit", default=None, type=int)
@click.option(
    "--face-model-name",
    default=None,
    help="InsightFace model pack for the identity validator (default: from config or 'antelopev2')",
)
def main(config_path: str, limit: int | None, face_model_name: str | None) -> None:
    cfg = load_config(config_path)

    # Validation outputs go to runs/<id>/validation/ — mirror the audit/analysis layout.
    validation_dir = Path(
        cfg.get("paths.validation_dir", str(Path(cfg["paths.output_dir"]) / "validation"))
    )
    validation_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest.create(
        run_id=cfg.get("run_id", "unknown") + "_validate",
        config_path=config_path,
        config_resolved=cfg.to_dict(),
    )

    generated_dir = Path(cfg["paths.generated_dir"])
    image_index = pd.read_parquet(generated_dir / "manifest.parquet")
    if limit is not None:
        image_index = image_index.head(limit)
    logger.info(f"Validating identity for {len(image_index)} counterfactuals")

    # Map seed_identity_id → seed_image_path. The seed sampler is deterministic
    # given the same config, so re-loading produces the same identities.
    si_cfg = cfg["seed_identities"]
    seeds = load_or_sample_seeds(
        output_dir=cfg["paths.seed_dataset"],
        n=int(si_cfg["count"]),
        stratify_by=list(si_cfg.get("stratify_by", ["race", "gender", "age"])),
        seed=cfg.get("seed", 42),
    )
    seed_paths = {s.id: s.image_path for s in seeds}

    val_cfg = cfg.get("validators", {}).get("identity", {})
    threshold = float(val_cfg.get("threshold", 0.5))
    model_name = face_model_name or cfg.get("generator.face_model_name", "antelopev2")

    validator = ArcFaceIdentityValidator(model_name=model_name, threshold=threshold)

    rows = []
    for _, row in tqdm(image_index.iterrows(), total=len(image_index)):
        seed_id = row["seed_identity_id"]
        seed_path = seed_paths.get(seed_id)
        cf_path = row["image_path"]
        if seed_path is None or not Path(seed_path).exists():
            logger.warning(f"Seed image missing for {seed_id} ({seed_path}); skipping")
            continue
        if not Path(cf_path).exists():
            logger.warning(f"Counterfactual missing: {cf_path}; skipping")
            continue
        score = validator.score_pair(seed_path, cf_path)
        rows.append({
            **{c: row[c] for c in image_index.columns},
            "seed_image_path": seed_path,
            "cosine_similarity": score.cosine_similarity,
            "is_preserved": score.is_preserved,
        })

    out_path = validation_dir / "identity_scores.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)

    summary = {
        "n_pairs": len(df),
        "n_preserved": int(df["is_preserved"].sum()) if len(df) else 0,
        "fraction_preserved": float(df["is_preserved"].mean()) if len(df) else 0.0,
        "mean_cosine_similarity": float(df["cosine_similarity"].mean()) if len(df) else None,
        "median_cosine_similarity": float(df["cosine_similarity"].median()) if len(df) else None,
        "threshold": threshold,
        "face_model_name": model_name,
    }
    (validation_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    manifest.finish()
    manifest.write(validation_dir / "run_manifest.json")
    logger.info(
        f"Validated {len(df)} pairs. "
        f"Mean cosine sim: {summary['mean_cosine_similarity']:.3f if summary['mean_cosine_similarity'] is not None else 'n/a'}, "
        f"fraction preserved (≥{threshold}): {summary['fraction_preserved']:.1%}. "
        f"Outputs: {validation_dir}"
    )


if __name__ == "__main__":
    main()
