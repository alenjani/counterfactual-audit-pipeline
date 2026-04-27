"""CLI: generate counterfactual images from seed identities or text prompts."""
from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from cap.data import load_or_sample_seeds
from cap.generator import FluxPuLIDControlNetGenerator, GenerationRequest
from cap.utils import RunManifest, get_logger, load_config, set_global_seed

logger = get_logger()


@click.command()
@click.option("--config", "config_path", required=True, help="Path to config YAML")
@click.option("--limit", default=None, type=int, help="Limit to first N seed identities (for debugging)")
def main(config_path: str, limit: int | None) -> None:
    cfg = load_config(config_path)
    set_global_seed(cfg.get("seed", 42))

    generated_dir = Path(cfg["paths.generated_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest.create(
        run_id=cfg.get("run_id", "unknown"),
        config_path=config_path,
        config_resolved=cfg.to_dict(),
    )

    gen_cfg = cfg["generator"]
    generator = FluxPuLIDControlNetGenerator(
        device=gen_cfg.get("device", "cuda"),
        dtype=gen_cfg.get("dtype", "fp8"),
    )
    manifest.model_versions = generator.model_versions()

    seeds = _load_seed_identities(cfg)
    if limit is not None:
        seeds = seeds[:limit]
    logger.info(f"Generating counterfactuals for {len(seeds)} seed identities")

    all_results = []
    for seed in seeds:
        request = GenerationRequest(
            seed_identity_id=seed["id"],
            seed_image_path=seed["image_path"],
            seed_prompt=None,
            counterfactual_axes=cfg["counterfactual_axes"],
            fixed_attributes=cfg.get("fixed_attributes", {}),
            seed=cfg.get("seed", 42),
            num_inference_steps=gen_cfg.get("num_inference_steps", 28),
            guidance_scale=gen_cfg.get("guidance_scale", 3.5),
            width=gen_cfg.get("width", 1024),
            height=gen_cfg.get("height", 1024),
        )
        results = generator.generate(request, generated_dir)
        all_results.extend([_result_to_dict(r) for r in results])

    output_index = generated_dir / "manifest.parquet"
    pd.DataFrame(all_results).to_parquet(output_index)
    manifest.finish()
    manifest.write(generated_dir / "run_manifest.json")
    logger.info(f"Generated {len(all_results)} images. Manifest: {output_index}")


def _load_seed_identities(cfg) -> list[dict]:
    """Load (or sample) FairFace seed identities per config."""
    si_cfg = cfg["seed_identities"]
    if si_cfg.get("source") != "fairface":
        raise ValueError(f"Unsupported seed source: {si_cfg.get('source')}. Only 'fairface' is wired.")
    seeds = load_or_sample_seeds(
        output_dir=cfg["paths.seed_dataset"],
        n=int(si_cfg["count"]),
        stratify_by=list(si_cfg.get("stratify_by", ["race", "gender", "age"])),
        seed=cfg.get("seed", 42),
    )
    return [{"id": s.id, "image_path": s.image_path,
             "race": s.race, "gender": s.gender, "age": s.age} for s in seeds]


def _result_to_dict(result) -> dict:
    return {
        "seed_identity_id": result.seed_identity_id,
        "counterfactual_id": result.counterfactual_id,
        "image_path": result.image_path,
        "prompt": result.prompt_used,
        **{f"axis_{k}": v for k, v in result.axis_values.items()},
    }


if __name__ == "__main__":
    main()
