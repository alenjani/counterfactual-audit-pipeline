"""Ray-based distributed runner for fan-out generation across multiple GPUs.

Wraps `FluxPuLIDControlNetGenerator` in a Ray actor (one per GPU) and uses
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
        dtype: str = "nf4",
        controlnet_mode: str = "pose",
        cache_dir: str = "/local_disk0/hf_cache",
        hf_token: str | None = None,
        face_model_name: str = "antelopev2",
        pulid_src: str | None = None,
    ):
        import os
        import sys

        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        if pulid_src and pulid_src not in sys.path:
            sys.path.insert(0, pulid_src)

        from cap.generator import FluxPuLIDControlNetGenerator

        self.generator = FluxPuLIDControlNetGenerator(
            dtype=dtype,
            controlnet_mode=controlnet_mode,
            cache_dir=cache_dir,
            face_model_name=face_model_name,
        )

    def warm(self) -> dict[str, Any]:
        """Force model load. Called once per actor before the first generate."""
        import time

        import torch

        t0 = time.time()
        self.generator._lazy_load()
        load_s = time.time() - t0
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
        return [
            {
                "seed_identity_id": r.seed_identity_id,
                "counterfactual_id": r.counterfactual_id,
                "image_path": r.image_path,
                "prompt_used": r.prompt_used,
                "axis_values": r.axis_values,
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

    Idempotency: the underlying `FluxPuLIDControlNetGenerator.generate` skips
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
