"""CLI: generate counterfactual images from seed identities or text prompts.

Two backends:
  - `local`  — single-process loop on the current GPU. Fine for the smoke and
               for sub-MVP scale (≤50 images). Defaults to backend=local.
  - `ray`    — fan out across N Ray actors (one per GPU). Use for MVP and full
               runs. Each actor holds Flux + ControlNet + PuLID + InsightFace
               in VRAM and pulls requests from a shared queue.

Both backends produce idempotent output: re-running with the same config and
output_dir skips counterfactuals whose PNGs already exist (resume-from-crash
is free).
"""
from __future__ import annotations

import json
import os
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
@click.option(
    "--backend",
    type=click.Choice(["local", "ray"]),
    default="local",
    help="Generation backend. local=single GPU, ray=multi-GPU fan-out.",
)
@click.option(
    "--num-actors",
    default=4,
    type=int,
    help="Ray fan-out width (only used when --backend=ray)",
)
@click.option(
    "--hf-token",
    default=None,
    help="HF token for gated repos. If unset, falls back to HF_TOKEN env var.",
)
@click.option(
    "--pulid-src",
    default=None,
    help="Path to PuLID source dir (sys.path-imported by actors). Only for --backend=ray.",
)
@click.option(
    "--cache-dir",
    default=None,
    help="HF model cache dir. Override the per-actor default for Volume-backed cache.",
)
@click.option(
    "--priority-mode",
    type=click.Choice(["none", "paper1_first"]),
    default="none",
    help=(
        "Order of generated cells. 'none' = legacy (one request per identity, "
        "all axes generated together, single seed). 'paper1_first' = expand to "
        "per-cell requests across all generation_seeds, sorted so Paper 1 cells "
        "(seed=42, anchor age) come first; required for staged full-instrument runs."
    ),
)
def main(
    config_path: str,
    limit: int | None,
    backend: str,
    num_actors: int,
    hf_token: str | None,
    pulid_src: str | None,
    cache_dir: str | None,
    priority_mode: str,
) -> None:
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
    seeds = _load_seed_identities(cfg)
    if limit is not None:
        seeds = seeds[:limit]
    logger.info(f"Generating counterfactuals for {len(seeds)} seed identities (backend={backend})")

    # Build the request list once — both backends consume the same shape.
    if priority_mode == "paper1_first":
        requests = _build_priority_requests(seeds, cfg, gen_cfg)
        logger.info(
            f"Priority mode 'paper1_first' → {len(requests)} per-cell requests "
            f"(was {len(seeds)} per-identity requests in legacy mode)"
        )
    else:
        requests = [_seed_to_request(s, cfg, gen_cfg) for s in seeds]

    if backend == "local":
        generator = FluxPuLIDControlNetGenerator(
            device=gen_cfg.get("device", "cuda"),
            dtype=gen_cfg.get("dtype", "fp8"),
            cache_dir=cache_dir,
            controlnet_mode=gen_cfg.get("controlnet_mode", "pose"),
            face_model_name=gen_cfg.get("face_model_name", "antelopev2"),
        )
        manifest.model_versions = generator.model_versions()

        all_results: list[dict] = []
        for request in requests:
            results = generator.generate(request, generated_dir)
            all_results.extend(_result_to_dict(r) for r in results)

    elif backend == "ray":
        import ray

        from cap.generator.ray_runner import run_distributed

        actor_kwargs = {
            "dtype": gen_cfg.get("dtype", "nf4"),
            "controlnet_mode": gen_cfg.get("controlnet_mode", "pose"),
            "cache_dir": cache_dir or "/local_disk0/hf_cache",
            "hf_token": hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            "face_model_name": gen_cfg.get("face_model_name", "antelopev2"),
            "pulid_src": pulid_src,
            "id_weight": float(gen_cfg.get("id_weight", 1.0)),
            "controlnet_conditioning_scale": float(gen_cfg.get("controlnet_conditioning_scale", 0.6)),
        }

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        n_avail = int(ray.cluster_resources().get("GPU", 0))
        if n_avail < num_actors:
            logger.warning(
                f"Requested {num_actors} Ray actors but only {n_avail} GPUs available — "
                f"reducing to {n_avail}."
            )
            num_actors = max(1, n_avail)

        # Stream the incremental manifest to /local_disk0 (driver-side) — the
        # Volume's GCS-FUSE backing fails on file-append (Errno 95). The PNGs
        # written by each actor (single-shot writes) ARE the durable source of
        # truth, and skip-if-exists handles resume-from-crash regardless of
        # whether the streaming manifest survived.
        local_manifest = Path("/local_disk0/cap_mvp_manifest.jsonl")
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        if local_manifest.exists():
            local_manifest.unlink()  # fresh file per run

        all_results = list(
            run_distributed(
                requests=requests,
                output_dir=str(generated_dir),
                num_actors=num_actors,
                actor_kwargs=actor_kwargs,
                manifest_path=str(local_manifest),
            )
        )

        # Copy the streaming manifest to the Volume at end (single-shot write).
        try:
            (generated_dir / "incremental_manifest.jsonl").write_bytes(local_manifest.read_bytes())
        except Exception as e:
            logger.warning(f"Manifest copy to Volume failed: {e}; PNGs + parquet still authoritative.")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    output_index = generated_dir / "manifest.parquet"
    pd.DataFrame(all_results).to_parquet(output_index)
    manifest.finish()
    manifest.write(generated_dir / "run_manifest.json")
    logger.info(f"Generated {len(all_results)} images. Manifest: {output_index}")


def _seed_to_request(seed: dict, cfg, gen_cfg) -> GenerationRequest:
    return GenerationRequest(
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


def _build_priority_requests(seeds: list[dict], cfg, gen_cfg) -> list[GenerationRequest]:
    """Per-cell requests sorted so Paper 1 (Stage A) cells come first.

    Stage A = (seed=42, age=anchor) — fully sufficient for Paper 1 ISR.
    Stage B = (seed=42, age!=anchor) — adds the age axis.
    Stage C = (seed!=42) — adds the noise-robustness check across all cells.

    Within each stage, cells are ordered by (identity, skin, gender, age) so
    skip-if-exists resume picks up at a deterministic point. See
    plans/003-full-instrument-plan.md for the cumulative staging rationale.
    """
    import itertools

    axes = cfg["counterfactual_axes"]
    skin_vals = axes.get("skin_tone") or [None]
    gender_vals = axes.get("gender") or [None]
    age_vals = axes.get("age") or [None]
    gen_seeds = cfg.get("generation_seeds", [cfg.get("seed", 42)])
    if not gen_seeds:
        gen_seeds = [42]
    anchor_age = cfg.get("priority_age_anchor", 40)

    items: list[tuple[tuple, GenerationRequest]] = []
    for s in seeds:
        for skin, gender, age, gen_seed in itertools.product(
            skin_vals, gender_vals, age_vals, gen_seeds
        ):
            single_axes: dict[str, list] = {}
            if skin is not None:
                single_axes["skin_tone"] = [skin]
            if gender is not None:
                single_axes["gender"] = [gender]
            if age is not None:
                single_axes["age"] = [age]

            # Stage tier (lower = earlier):
            #   tier 0 = seed=42 + age=anchor (Stage A: Paper 1)
            #   tier 1 = seed=42 + age!=anchor (Stage B)
            #   tier 2 = seed!=42 (Stage C)
            if gen_seed == gen_seeds[0]:
                tier = 0 if (age is None or age == anchor_age) else 1
            else:
                tier = 2 + gen_seeds.index(gen_seed) - 1
            priority = (tier, s["id"], skin, gender, age, gen_seed)

            req = GenerationRequest(
                seed_identity_id=s["id"],
                seed_image_path=s["image_path"],
                seed_prompt=None,
                counterfactual_axes=single_axes,
                fixed_attributes=cfg.get("fixed_attributes", {}),
                seed=int(gen_seed),
                num_inference_steps=gen_cfg.get("num_inference_steps", 28),
                guidance_scale=gen_cfg.get("guidance_scale", 3.5),
                width=gen_cfg.get("width", 1024),
                height=gen_cfg.get("height", 1024),
            )
            items.append((priority, req))

    items.sort(key=lambda x: tuple(str(v) if v is not None else "" for v in x[0]))
    return [r for _, r in items]


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
