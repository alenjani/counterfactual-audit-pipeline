"""Flux.1 Dev + PuLID + ControlNet counterfactual generator (L4 / 24 GB).

Architecture
------------
1. **Flux.1 Dev** as the base diffusion model (12B-parameter DiT). Quantized to
   FP8 (or NF4) so the full pipeline fits in 24 GB alongside ControlNet + PuLID.
2. **ControlNet-Union** (InstantX/FLUX.1-dev-Controlnet-Union) for structural
   conditioning — locks pose, head angle, lighting via a control image.
3. **PuLID** for identity preservation — extracts an identity embedding from
   the seed face (via InsightFace ArcFace) and injects it through the PuLID
   adapter so the generated face matches the seed identity while the demographic
   prompt varies.

PuLID integration note
----------------------
There are two viable PuLID-for-Flux integration paths:

  (a) **Official `pulid` package** (`pip install pulid` once PR merges, currently
      via `git+https://github.com/ToTheBeginning/PuLID`) — patches Flux attention.
  (b) **IP-Adapter FaceID Plus** — a more standardized identity injection path
      (`h94/IP-Adapter-FaceID`) that loads cleanly into diffusers.

We default to (a) when available, fall back to (b) when not. Both use ArcFace
embeddings from InsightFace as input, so the upstream flow is identical.

Memory budget on L4 (24 GB)
---------------------------
- Flux.1 Dev (FP8/NF4):  ~14–16 GB
- ControlNet:             ~3 GB
- PuLID encoder + IP head: ~1 GB
- Activations (1024²):    ~3 GB
- Headroom:               ~1–3 GB
Tight but workable. If OOM: drop to NF4 or 768² resolution.
"""
from __future__ import annotations

import gc
import itertools
import time
from pathlib import Path
from typing import Any

from PIL import Image

from cap.generator.base import CounterfactualGenerator, GenerationRequest, GenerationResult
from cap.generator.prompts import build_demographic_prompt
from cap.utils.logging import get_logger

logger = get_logger()


class FluxPuLIDControlNetGenerator(CounterfactualGenerator):
    def __init__(
        self,
        flux_model_id: str = "black-forest-labs/FLUX.1-dev",
        controlnet_model_id: str = "InstantX/FLUX.1-dev-Controlnet-Union",
        pulid_model_id: str = "guozinan/PuLID",
        ip_adapter_id: str = "h94/IP-Adapter-FaceID",
        device: str = "cuda",
        dtype: str = "fp8",
        cache_dir: str | None = None,
        identity_path: str = "pulid",  # "pulid" | "ip_adapter" | "none"
        controlnet_mode: str = "pose",  # ControlNet-Union mode index
    ):
        self.flux_model_id = flux_model_id
        self.controlnet_model_id = controlnet_model_id
        self.pulid_model_id = pulid_model_id
        self.ip_adapter_id = ip_adapter_id
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.identity_path = identity_path
        self.controlnet_mode = controlnet_mode

        self._pipeline = None
        self._pulid = None
        self._ip_adapter_loaded = False
        self._face_app = None  # InsightFace for ID embedding
        self._control_processor = None

    # ------------------------------------------------------------------ loading

    def _torch_dtype(self):
        import torch

        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
            self.dtype, torch.bfloat16
        )

    def _quant_config(self):
        """Quantization config for L4 fit. FP8 preferred; NF4 as smaller fallback.

        diffusers >=0.32 requires PipelineQuantizationConfig wrapping the
        BitsAndBytesConfig per-component, not a raw BitsAndBytesConfig.
        """
        if self.dtype not in {"fp8", "nf4"}:
            return None
        from diffusers import PipelineQuantizationConfig

        if self.dtype == "fp8":
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=["transformer", "text_encoder_2"],
            )
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
            components_to_quantize=["transformer", "text_encoder_2"],
        )

    def _lazy_load(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from diffusers import FluxControlNetModel, FluxControlNetPipeline

        logger.info(
            f"Loading Flux pipeline | flux={self.flux_model_id} | "
            f"controlnet={self.controlnet_model_id} | dtype={self.dtype}"
        )
        t0 = time.time()

        torch_dtype = self._torch_dtype()
        quant = self._quant_config()

        controlnet = FluxControlNetModel.from_pretrained(
            self.controlnet_model_id, torch_dtype=torch_dtype, cache_dir=self.cache_dir
        )

        pipe_kwargs: dict[str, Any] = {
            "controlnet": controlnet,
            "torch_dtype": torch_dtype,
            "cache_dir": self.cache_dir,
        }
        if quant is not None:
            pipe_kwargs["quantization_config"] = quant

        self._pipeline = FluxControlNetPipeline.from_pretrained(self.flux_model_id, **pipe_kwargs)

        # Memory-saving offload (CPU offload for non-active modules) — important on L4
        self._pipeline.enable_model_cpu_offload(device=self.device)
        try:
            self._pipeline.vae.enable_tiling()
        except Exception:
            pass

        # Identity injection setup
        if self.identity_path == "pulid":
            self._load_pulid()
        elif self.identity_path == "ip_adapter":
            self._load_ip_adapter()

        # Structural preprocessor (pose / depth) for ControlNet
        from cap.generator.structural import build_control_processor

        self._control_processor = build_control_processor(self.controlnet_mode, device=self.device)

        # Face embedding extractor (used by both PuLID and IP-Adapter paths).
        # Explicit `root` is required because insightface's default `~/.insightface/`
        # is not consistently writable across environments (Databricks workers,
        # ephemeral containers); a missing/empty model dir causes FaceAnalysis to
        # init with an empty self.models and fail `assert 'detection' in self.models`.
        from insightface.app import FaceAnalysis

        insightface_root = self.cache_dir or "/tmp/insightface_models"
        Path(insightface_root).mkdir(parents=True, exist_ok=True)

        self._face_app = FaceAnalysis(
            name="antelopev2",
            root=insightface_root,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

        logger.info(f"Pipeline loaded in {time.time() - t0:.1f}s")

    def _load_pulid(self) -> None:
        """Load PuLID-Flux. Requires the PuLID package — install instructions in README."""
        try:
            from pulid.pipeline_flux import PuLIDPipeline  # type: ignore

            self._pulid = PuLIDPipeline(self._pipeline, device=self.device)
            self._pulid.load_pretrain(self.pulid_model_id)
            logger.info("PuLID-Flux adapter loaded")
        except ImportError:
            logger.warning(
                "PuLID package not installed — falling back to IP-Adapter FaceID for identity. "
                "To enable PuLID: pip install git+https://github.com/ToTheBeginning/PuLID"
            )
            self.identity_path = "ip_adapter"
            self._load_ip_adapter()

    def _load_ip_adapter(self) -> None:
        """Load IP-Adapter FaceID as identity-injection fallback."""
        try:
            self._pipeline.load_ip_adapter(
                self.ip_adapter_id,
                subfolder=None,
                weight_name="ip-adapter-faceid_flux.bin",
            )
            self._pipeline.set_ip_adapter_scale(0.7)
            self._ip_adapter_loaded = True
            logger.info("IP-Adapter FaceID loaded for identity preservation")
        except Exception as e:
            logger.error(f"IP-Adapter load failed: {e}. Identity preservation will be weak.")
            self.identity_path = "none"

    # --------------------------------------------------------------- generation

    def _extract_id_embedding(self, image_path: str | Path):
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read seed image: {image_path}")
        faces = self._face_app.get(img)
        if not faces:
            raise ValueError(f"No face detected in seed image: {image_path}")
        return faces[0].normed_embedding.astype(np.float32)

    def _build_control_image(self, seed_image: Image.Image) -> Image.Image:
        return self._control_processor(seed_image)

    def generate(
        self, request: GenerationRequest, output_dir: str | Path
    ) -> list[GenerationResult]:
        import torch

        self._lazy_load()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        seed_img = Image.open(request.seed_image_path).convert("RGB") if request.seed_image_path else None
        control_image = self._build_control_image(seed_img) if seed_img is not None else None

        id_embedding = None
        if self.identity_path != "none" and request.seed_image_path:
            id_embedding = self._extract_id_embedding(request.seed_image_path)

        results: list[GenerationResult] = []
        axes = request.counterfactual_axes
        keys = list(axes.keys())

        for combo in itertools.product(*[axes[k] for k in keys]):
            axis_values = dict(zip(keys, combo))
            prompt = build_demographic_prompt(
                base_attributes=request.fixed_attributes,
                demographic_attributes=axis_values,
            )
            counterfactual_id = self._make_id(request.seed_identity_id, axis_values)
            image_path = output_dir / f"{counterfactual_id}.png"

            t0 = time.time()
            generator = torch.Generator(device=self.device).manual_seed(request.seed)

            pipe_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "width": request.width,
                "height": request.height,
                "generator": generator,
            }
            if control_image is not None:
                pipe_kwargs["control_image"] = control_image
                pipe_kwargs["controlnet_conditioning_scale"] = 0.6
            if self.identity_path == "ip_adapter" and id_embedding is not None:
                pipe_kwargs["ip_adapter_image_embeds"] = [
                    torch.from_numpy(id_embedding).unsqueeze(0).to(self.device)
                ]

            if self.identity_path == "pulid" and self._pulid is not None and id_embedding is not None:
                # PuLID has its own forward — bypasses standard pipe_kwargs
                image = self._pulid.generate(
                    prompt=prompt,
                    id_embedding=id_embedding,
                    control_image=control_image,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    width=request.width,
                    height=request.height,
                    generator=generator,
                )
            else:
                image = self._pipeline(**pipe_kwargs).images[0]

            image.save(image_path)
            elapsed = time.time() - t0

            results.append(
                GenerationResult(
                    seed_identity_id=request.seed_identity_id,
                    counterfactual_id=counterfactual_id,
                    image_path=str(image_path),
                    prompt_used=prompt,
                    axis_values=axis_values,
                    metadata={
                        "seed": request.seed,
                        "dtype": self.dtype,
                        "identity_path": self.identity_path,
                        "elapsed_s": round(elapsed, 2),
                    },
                )
            )
            logger.debug(f"  {counterfactual_id} in {elapsed:.1f}s")

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        return results

    @staticmethod
    def _make_id(seed_id: str, axis_values: dict[str, Any]) -> str:
        parts = [seed_id]
        for k in sorted(axis_values.keys()):
            parts.append(f"{k}{axis_values[k]}")
        return "_".join(str(p) for p in parts)

    def model_versions(self) -> dict[str, str]:
        return {
            "flux": self.flux_model_id,
            "controlnet": self.controlnet_model_id,
            "pulid": self.pulid_model_id if self.identity_path == "pulid" else "(unused)",
            "ip_adapter": self.ip_adapter_id if self.identity_path == "ip_adapter" else "(unused)",
            "dtype": self.dtype,
            "identity_path": self.identity_path,
            "controlnet_mode": self.controlnet_mode,
        }
