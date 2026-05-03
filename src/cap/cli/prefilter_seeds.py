"""CLI: prefilter FairFace seed identities for full-run readiness.

3-step filter (per plans/003-full-instrument-plan.md, "Pre-generation seed
filter" section):

  Step 0 — Load N stratified candidates (oversample by ~25% over the target).
  Step 1 — antelopev2 face-detection prefilter: drop seeds with `n_faces ≥ 2`
            (multi-face seed images). Catches ~60% of deterministic failures.
  Step 2 — Single-cell round-trip generation: for each survivor, generate ONE
            cell (skin=anchor, gender=preserve, age=anchor) at seed=42, compute
            ArcFace cosine vs the seed image, drop seeds whose cosine < threshold.
            Catches the remaining ~40% of deterministic failures.

Output: JSON list of confirmed seed IDs at `<output_dir>/confirmed_seeds.json`.
The list is consumed by `cap-generate ... --seed-ids-file <path>` for the
production runs.

Usage:
  cap-prefilter-seeds \
    --config configs/full.yaml \
    --target-confirmed 200 \
    --candidates 250 \
    --output-dir /Volumes/.../runs/full/seed_filter
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd

from cap.data import load_or_sample_seeds
from cap.utils import RunManifest, get_logger, load_config

logger = get_logger()


@click.command()
@click.option("--config", "config_path", required=True, help="Path to config YAML")
@click.option("--candidates", default=250, type=int, help="Number of candidate seeds to load (oversample)")
@click.option("--target-confirmed", default=200, type=int, help="Target final confirmed-seed count (informational)")
@click.option("--threshold", default=0.5, type=float, help="ArcFace cosine threshold for round-trip validation")
@click.option("--output-dir", required=True, help="Where to write seed_filter/ outputs")
@click.option("--cache-dir", default=None, help="HF cache for antelopev2 + Flux models")
@click.option("--pulid-src", default=None, help="PuLID source dir (sys.path-imported)")
@click.option("--hf-token", default=None, help="HF token for gated repos")
@click.option(
    "--mode",
    type=click.Choice(["faces_only", "round_trip", "full"]),
    default="full",
    help=(
        "faces_only: just the multi-face filter (Step 1) — quick. "
        "round_trip: just Step 2 (assumes face filter already done; reads candidates from <output_dir>/face_passed.json). "
        "full: both steps in sequence."
    ),
)
def main(
    config_path: str,
    candidates: int,
    target_confirmed: int,
    threshold: float,
    output_dir: str,
    cache_dir: str | None,
    pulid_src: str | None,
    hf_token: str | None,
    mode: str,
) -> None:
    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest.create(
        run_id=f"{cfg.get('run_id','prefilter')}_seed_prefilter",
        config_path=config_path,
        config_resolved=cfg.to_dict(),
    )

    # ---- Step 0: load candidates ---------------------------------------------
    seeds = load_or_sample_seeds(
        output_dir=cfg["paths.seed_dataset"],
        n=candidates,
        stratify_by=list(cfg["seed_identities"].get("stratify_by", ["race", "gender", "age"])),
        seed=cfg.get("seed", 42),
    )
    logger.info(f"Step 0: loaded {len(seeds)} candidate seeds (target ≥ {target_confirmed} after filter)")

    # ---- Step 1: face-detection prefilter ------------------------------------
    if mode in ("faces_only", "full"):
        logger.info("Step 1: antelopev2 face-detection prefilter (drop n_faces >= 2)")
        face_passed_ids = _step1_face_filter(seeds, cache_dir, hf_token)
        face_passed_path = out_dir / "face_passed.json"
        face_passed_path.write_text(json.dumps(sorted(face_passed_ids)))
        logger.info(
            f"Step 1 done: {len(face_passed_ids)} of {len(seeds)} candidates passed "
            f"({100*len(face_passed_ids)/len(seeds):.1f}%); saved to {face_passed_path}"
        )
        if mode == "faces_only":
            manifest.finish()
            manifest.write(out_dir / "step1_manifest.json")
            return
        survivors = [s for s in seeds if s.id in face_passed_ids]
    else:
        # round_trip mode: read prior face-filter output
        face_passed_path = out_dir / "face_passed.json"
        if not face_passed_path.exists():
            raise click.ClickException(
                f"--mode round_trip requires {face_passed_path} from a prior --mode faces_only run"
            )
        face_passed_ids = set(json.loads(face_passed_path.read_text()))
        survivors = [s for s in seeds if s.id in face_passed_ids]
        logger.info(f"Step 1 (re-loaded): {len(survivors)} survivors from disk")

    # ---- Step 2: round-trip validation --------------------------------------
    logger.info(
        f"Step 2: round-trip generate ONE cell per survivor + ArcFace cosine "
        f"(threshold={threshold})"
    )
    confirmed_ids, scores = _step2_round_trip(
        survivors, cfg, cache_dir, hf_token, pulid_src, threshold, out_dir
    )
    logger.info(
        f"Step 2 done: {len(confirmed_ids)} of {len(survivors)} survivors confirmed "
        f"({100*len(confirmed_ids)/len(survivors):.1f}%)"
    )

    # ---- Write final outputs -------------------------------------------------
    confirmed_path = out_dir / "confirmed_seeds.json"
    confirmed_path.write_text(json.dumps(sorted(confirmed_ids)))
    logger.info(f"Confirmed seeds list → {confirmed_path}")

    pd.DataFrame(scores).to_parquet(out_dir / "round_trip_scores.parquet")

    manifest.finish()
    manifest.write(out_dir / "prefilter_manifest.json")

    if len(confirmed_ids) < target_confirmed:
        logger.warning(
            f"WARNING: confirmed seeds ({len(confirmed_ids)}) < target ({target_confirmed}). "
            f"Consider re-running with --candidates {int(candidates * 1.2)} to oversample further."
        )


def _step1_face_filter(seeds: list, cache_dir: str | None, hf_token: str | None) -> set:
    """Run antelopev2 face detection on each seed; return IDs with exactly 1 face."""
    import cv2
    from PIL import Image
    from insightface.app import FaceAnalysis

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if cache_dir:
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)

    app = FaceAnalysis(
        name="antelopev2",
        root=str(Path("/local_disk0/pulid_workdir")) if Path("/local_disk0").exists() else ".",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    keep: set[str] = set()
    for s in seeds:
        try:
            img = np.asarray(Image.open(s.image_path).convert("RGB"))[:, :, ::-1]  # BGR
            faces = app.get(img)
            if len(faces) == 1:
                keep.add(s.id)
            else:
                logger.info(f"  drop {s.id}: n_faces={len(faces)}")
        except Exception as e:
            logger.warning(f"  drop {s.id}: face-detect error {e}")
    return keep


def _step2_round_trip(
    survivors: list,
    cfg,
    cache_dir: str | None,
    hf_token: str | None,
    pulid_src: str | None,
    threshold: float,
    out_dir: Path,
) -> tuple[set, list[dict]]:
    """For each survivor, generate one cell + score cosine vs seed."""
    import sys

    # Set up env exactly like the Ray actor does (lifted from ray_runner.py).
    local_hf_home = "/local_disk0/hf_cache"
    Path(local_hf_home).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = local_hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = local_hf_home
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/dev/shm/torch_extensions")
    os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
    Path("/dev/shm/torch_extensions").mkdir(parents=True, exist_ok=True)
    Path("/dev/shm/tmp").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if pulid_src and pulid_src not in sys.path:
        sys.path.insert(0, pulid_src)

    # chdir for PuLID's relative-path writes (mirror ray_runner.warm()).
    workdir = Path("/local_disk0/pulid_workdir")
    workdir.mkdir(parents=True, exist_ok=True)
    saved_cwd = os.getcwd()
    os.chdir(workdir)

    try:
        from cap.generator import FluxPuLIDNativeGenerator, GenerationRequest
        from cap.validator import ArcFaceIdentityValidator

        gen_cfg = cfg["generator"]
        gen = FluxPuLIDNativeGenerator(
            flux_model_name="flux-dev",
            controlnet_mode=gen_cfg.get("controlnet_mode", "pose"),
            cache_dir=local_hf_home,
            face_model_name=gen_cfg.get("face_model_name", "antelopev2"),
            use_fp8=True,
            id_weight=float(gen_cfg.get("id_weight", 1.5)),
            controlnet_conditioning_scale=float(gen_cfg.get("controlnet_conditioning_scale", 0.6)),
        )

        gen_dir = out_dir / "round_trip_pngs"
        gen_dir.mkdir(parents=True, exist_ok=True)

        anchor_age = cfg.get("priority_age_anchor", 40)
        anchor_axes = {"skin_tone": [3], "gender": ["male"]}
        if "age" in cfg.get("counterfactual_axes", {}):
            anchor_axes["age"] = [anchor_age]

        validator = ArcFaceIdentityValidator(
            model_name="antelopev2", threshold=threshold, root=local_hf_home
        )
        validator._lazy_load()

        confirmed: set[str] = set()
        scores: list[dict] = []
        for s in survivors:
            try:
                req = GenerationRequest(
                    seed_identity_id=s.id,
                    seed_image_path=s.image_path,
                    seed_prompt=None,
                    counterfactual_axes=anchor_axes,
                    fixed_attributes=cfg.get("fixed_attributes", {}),
                    seed=42,
                    num_inference_steps=int(gen_cfg.get("num_inference_steps", 40)),
                    guidance_scale=float(gen_cfg.get("guidance_scale", 3.5)),
                    width=int(gen_cfg.get("width", 768)),
                    height=int(gen_cfg.get("height", 768)),
                )
                results = gen.generate(req, gen_dir)
                if not results:
                    scores.append({"id": s.id, "cosine": float("nan"), "confirmed": False, "reason": "no_result"})
                    continue
                e_seed = validator.embed(s.image_path)
                e_gen = validator.embed(results[0].image_path)
                if e_seed is None or e_gen is None:
                    scores.append({"id": s.id, "cosine": float("nan"), "confirmed": False, "reason": "embed_none"})
                    continue
                cos = float(np.dot(e_seed, e_gen))
                ok = cos >= threshold
                if ok:
                    confirmed.add(s.id)
                scores.append({"id": s.id, "cosine": cos, "confirmed": ok})
                logger.info(f"  {s.id}: cosine={cos:.4f} {'✓' if ok else '✗'}")
            except Exception as e:
                scores.append({"id": s.id, "cosine": float("nan"), "confirmed": False, "reason": str(e)[:200]})
                logger.warning(f"  {s.id}: round-trip error: {e}")
        return confirmed, scores
    finally:
        os.chdir(saved_cwd)


if __name__ == "__main__":
    main()
