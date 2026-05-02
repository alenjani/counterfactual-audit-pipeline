"""Ray-based distributed runner for fan-out generation across multiple GPUs.

Wraps `FluxPuLIDNativeGenerator` in a Ray actor (one per GPU) and uses
`ray.util.ActorPool.map_unordered` to distribute `GenerationRequest`s across
the actor pool. Each actor holds Flux + ControlNet + PuLID + InsightFace in
VRAM; requests are streamed in, images written to `output_dir`, results
yielded as they complete.

Used by `cap.cli.generate --backend ray` for MVP and full-scale runs. The
smoke notebook uses an inline FluxActor with the same shape — this module
is the productionized version of that prototype.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import ray

from cap.generator.base import GenerationRequest


def _request_to_dict(r: GenerationRequest) -> dict:
    """Serialize a GenerationRequest for actor remote() — Ray needs a picklable payload."""
    return {
        "seed_identity_id": r.seed_identity_id,
        "seed_image_path": r.seed_image_path,
        "seed_prompt": r.seed_prompt,
        "counterfactual_axes": r.counterfactual_axes,
        "fixed_attributes": r.fixed_attributes,
        "seed": r.seed,
        "num_inference_steps": r.num_inference_steps,
        "guidance_scale": r.guidance_scale,
        "width": r.width,
        "height": r.height,
    }


@ray.remote(num_gpus=1)
class FluxActor:
    """One-GPU actor that loads Flux+ControlNet+PuLID+InsightFace and serves generation requests."""

    def __init__(
        self,
        dtype: str = "fp8",  # informational only; FluxPuLIDNativeGenerator hardcodes FP8 path
        controlnet_mode: str = "pose",
        cache_dir: str = "/local_disk0/hf_cache",
        hf_token: str | None = None,
        face_model_name: str = "antelopev2",
        pulid_src: str | None = None,
    ):
        import os
        import sys
        from pathlib import Path

        # HF_HOME MUST be on a non-FUSE filesystem. HF Hub writes downloads to
        # `<HF_HOME>/.../<hash>.incomplete` then renames to the final path.
        # GCS-FUSE-backed Volumes silently drop the rename (the .incomplete
        # file vanishes mid-rename → FileNotFoundError on copy fallback).
        # /local_disk0 is per-worker local SSD — fast, writable, non-FUSE.
        # The cache_dir argument is honored separately for diffusers'
        # from_pretrained calls where it's safe to pass.
        local_hf_home = "/local_disk0/hf_cache"
        Path(local_hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = local_hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = local_hf_home
        os.environ["TRANSFORMERS_CACHE"] = local_hf_home
        # HF Hub's xet downloader fails on GCS-FUSE Volumes ("File too large,
        # os error 27"). Force plain HTTP download until xet supports FUSE.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        # Faster downloads of large weight files (12 GB FP8 Flux etc).
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        # quanto JIT-compile cache: /dev/shm (32 GB tmpfs) avoids /local_disk0
        # filling up after Flux + PuLID + antelopev2 downloads.
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/dev/shm/torch_extensions")
        os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
        Path("/dev/shm/torch_extensions").mkdir(parents=True, exist_ok=True)
        Path("/dev/shm/tmp").mkdir(parents=True, exist_ok=True)
        # Reduce CUDA fragmentation on the L4.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # `cache_dir` is intentionally NOT used as HF_HOME — it goes to
        # diffusers' from_pretrained calls where Volume paths can work
        # because we set HF_HUB_DISABLE_XET=1.
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        if pulid_src and pulid_src not in sys.path:
            sys.path.insert(0, pulid_src)

        from cap.generator import FluxPuLIDNativeGenerator

        # FluxPuLIDNativeGenerator (the architectural fix validated 2026-05-01)
        # uses PuLID's native Flux + a monkey-patched forward that honors BOTH
        # pulid_ca id-injection AND diffusers ControlNet residuals. Replaces
        # the broken FluxPuLIDControlNetGenerator that silently fell back to
        # text-only Flux because diffusers' FluxTransformer2DModel doesn't
        # honor pulid_ca attributes (Review 001 §4 / preserved_fraction = 0).
        self.generator = FluxPuLIDNativeGenerator(
            flux_model_name="flux-dev",
            controlnet_mode=controlnet_mode,
            # Use local SSD for the HF cache too — diffusers' from_pretrained
            # has the same .incomplete-rename issue on GCS-FUSE Volumes that
            # caused the original MVP host to OOM. Each worker is a separate
            # node with its own /local_disk0, so cross-worker re-download is
            # unavoidable but cheap (~5 min/12 GB with HF_HUB_ENABLE_HF_TRANSFER).
            cache_dir=local_hf_home,
            face_model_name=face_model_name,
            use_fp8=True,
        )

    def warm(self) -> dict[str, Any]:
        """Force model load. Called once per actor before the first generate."""
        import os
        import time
        from pathlib import Path

        import torch

        # PuLID's load_pretrain (and PuLIDPipeline.__init__) use relative
        # `local_dir='models'` paths for hf_hub_download — they land in CWD.
        # Default CWD on a Ray actor is on a non-writable / FUSE filesystem
        # depending on cluster config; explicitly chdir to a /local_disk0
        # workdir so all PuLID-relative writes go somewhere fast and writable.
        pulid_workdir = Path("/local_disk0/pulid_workdir")
        pulid_workdir.mkdir(parents=True, exist_ok=True)
        saved_cwd = os.getcwd()
        os.chdir(pulid_workdir)
        try:
            t0 = time.time()
            self.generator._lazy_load()
            load_s = time.time() - t0
        finally:
            os.chdir(saved_cwd)

        vram_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        return {
            "load_s": round(load_s, 1),
            "vram_mb_after_load": round(vram_mb, 0),
            "model_versions": self.generator.model_versions(),
        }

    def generate(self, request_dict: dict, output_dir: str) -> list[dict]:
        """Generate counterfactuals for one identity. Returns list of result dicts.

        Idempotent at the per-image level — `FluxPuLIDControlNetGenerator.generate`
        skips counterfactual_id PNGs that already exist in output_dir, so re-running
        a partially-completed job picks up only what's missing.
        """
        from cap.generator.base import GenerationRequest

        request = GenerationRequest(**request_dict)
        results = self.generator.generate(request, output_dir)
        # Flatten axis_values into per-axis columns to match the local backend's
        # _result_to_dict shape. Otherwise downstream consumers (audit, analyze)
        # see a dict column they can't groupby/filter on.
        return [
            {
                "seed_identity_id": r.seed_identity_id,
                "counterfactual_id": r.counterfactual_id,
                "image_path": r.image_path,
                "prompt_used": r.prompt_used,
                **{f"axis_{k}": v for k, v in r.axis_values.items()},
                "metadata": r.metadata,
            }
            for r in results
        ]


def run_distributed(
    requests: list[GenerationRequest],
    output_dir: str,
    num_actors: int = 4,
    actor_kwargs: dict | None = None,
    manifest_path: str | None = None,
) -> Iterator[dict]:
    """Fan out a list of GenerationRequests across `num_actors` Ray actors.

    Yields per-image result dicts as they complete (out of order). If
    `manifest_path` is given, each result is also appended as a JSONL line
    to that file (durable, resumable across cluster crashes).

    Idempotency: the underlying `FluxPuLIDNativeGenerator.generate` skips
    images that already exist in `output_dir`, so re-running this with the
    same requests + output_dir is safe — only missing images get generated.
    """
    from ray.util import ActorPool

    actor_kwargs = dict(actor_kwargs or {})

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if manifest_path:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)

    actors = [FluxActor.remote(**actor_kwargs) for _ in range(num_actors)]

    # Warm in parallel — first actor pays the model-download tax, the rest
    # hit the (Volume-cached) HF dir and just load into their own VRAM.
    warm_infos = ray.get([a.warm.remote() for a in actors])
    for i, info in enumerate(warm_infos):
        # Use repr() to avoid bringing in cap's logger (which the caller may
        # already have configured differently).
        print(f"[ray_runner] actor {i} warm: {info['load_s']}s, "
              f"{info['vram_mb_after_load']:.0f} MB VRAM")

    pool = ActorPool(actors)
    request_dicts = [_request_to_dict(r) for r in requests]

    # Open-append-close per write (instead of a persistent handle) — GCS-FUSE
    # backed Volumes don't reliably support long-lived file handles across many
    # flush()/close() ops, raising OSError 95 ("operation not supported").
    # Per-write open is fine performance-wise: a few hundred small append writes
    # over the lifetime of a multi-hour run.
    def _append_manifest(result: dict) -> None:
        if not manifest_path:
            return
        with open(manifest_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    for batch in pool.map_unordered(
        lambda actor, req: actor.generate.remote(req, output_dir),
        request_dicts,
    ):
        for result in batch:
            _append_manifest(result)
            yield result
