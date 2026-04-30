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

# InstantX/FLUX.1-dev-Controlnet-Union mode index (required by the model's
# forward pass — passed as `control_mode` int alongside the control image).
_CONTROLNET_UNION_MODE_INDEX = {
    "canny": 0,
    "tile": 1,
    "depth": 2,
    "blur": 3,
    "pose": 4,
    "gray": 5,
    "lq": 6,
}


def _ensure_antelopev2(insightface_root: str) -> None:
    """Pre-stage InsightFace antelopev2 model files at <root>/models/antelopev2/.

    Why this exists: InsightFace's built-in auto-download silently fails in
    some environments (egress restrictions on its GitHub release URL, partial
    extracts that pass FaceAnalysis's assertion-only validation). We explicitly
    download the antelopev2 zip and extract it ourselves, with a HuggingFace
    mirror fallback so this works on locked-down clusters.

    antelopev2 is the higher-fidelity InsightFace pack (ArcFace ResNet-100 vs
    ResNet-50, WebFace42M training data) — required for the identity-preservation
    measurements that back CAP's statistical claims. See CLAUDE.md item 2.
    """
    import urllib.request
    import zipfile

    target_dir = Path(insightface_root) / "models" / "antelopev2"
    sentinel = target_dir / "scrfd_10g_bnkps.onnx"
    if sentinel.exists():
        logger.info(f"antelopev2 already present at {target_dir}")
        return

    parent = target_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    zip_path = parent / "antelopev2.zip"

    sources = [
        ("github_release", "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"),
        ("hf_mirror_monstermmorpg", "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip"),
    ]

    last_err: Exception | None = None
    for name, url in sources:
        try:
            logger.info(f"Downloading antelopev2 from {name}: {url}")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:
            logger.warning(f"antelopev2 source {name} failed: {e}")
            last_err = e
    else:
        raise RuntimeError(
            f"All antelopev2 sources failed; last error: {last_err}. "
            "Add another mirror to _ensure_antelopev2 or pre-stage the files manually."
        )

    # Extract. antelopev2.zip from the upstream release contains a top-level
    # `antelopev2/` folder; some HF mirrors host the files at the zip root.
    # Handle both layouts.
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(parent)
    zip_path.unlink()

    # Locate scrfd_10g_bnkps.onnx wherever it landed and move all sibling files
    # into target_dir so FaceAnalysis(name="antelopev2", root=...) finds them.
    candidates = list(parent.rglob("scrfd_10g_bnkps.onnx"))
    if not candidates:
        raise RuntimeError(
            f"antelopev2 zip extracted but scrfd_10g_bnkps.onnx not found under {parent}. "
            "The mirror's archive layout is different than expected."
        )
    src_dir = candidates[0].parent
    if src_dir.resolve() != target_dir.resolve():
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            f.rename(target_dir / f.name)
        if src_dir.exists() and not any(src_dir.iterdir()):
            src_dir.rmdir()

    files = sorted(p.name for p in target_dir.iterdir())
    logger.info(f"antelopev2 ready at {target_dir}: {files}")


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
        face_model_name: str = "antelopev2",  # InsightFace model pack: "antelopev2" (preferred) or "buffalo_l" (smaller fallback)
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
        self.face_model_name = face_model_name

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

        # antelopev2's auto-download fails on some clusters (egress block on
        # InsightFace's GitHub release URL). Pre-stage the files explicitly via
        # our own download (see _ensure_antelopev2). buffalo_l auto-downloads
        # cleanly so it doesn't need this.
        if self.face_model_name == "antelopev2":
            _ensure_antelopev2(insightface_root)

        # Diagnostic: check what's in the model dir before FaceAnalysis tries to load.
        # If FaceAnalysis fails the 'detection' assertion, this tells us whether the
        # auto-download even happened.
        import os as _os
        model_pack_dir = _os.path.join(insightface_root, "models", self.face_model_name)
        before = sorted(_os.listdir(model_pack_dir)) if _os.path.isdir(model_pack_dir) else "(missing)"
        logger.info(f"insightface model dir before load: {model_pack_dir} -> {before}")

        try:
            self._face_app = FaceAnalysis(
                name=self.face_model_name,
                root=insightface_root,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except AssertionError:
            after = sorted(_os.listdir(model_pack_dir)) if _os.path.isdir(model_pack_dir) else "(still missing)"
            raise RuntimeError(
                f"InsightFace failed to populate {model_pack_dir} (auto-download likely blocked). "
                f"Dir contents after attempt: {after}. "
                f"Either fix the cluster's egress to insightface release URLs, or pre-stage the "
                f"{self.face_model_name} ONNX files into {model_pack_dir} before calling generator."
            )
        self._face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

        logger.info(f"Pipeline loaded in {time.time() - t0:.1f}s")

    def _load_pulid(self) -> None:
        """Load PuLID-Flux. Requires the PuLID package — install instructions in README.

        Per CLAUDE.md item 2, when identity_path="pulid" is requested explicitly the
        generator must NOT silently fall back to IP-Adapter on failure. Such fallback
        was the root cause of the MVP run's preserved_fraction=0.0 (PuLID's eva_clip
        module needs ftfy/regex/torchsde, which weren't in cap's deps; the import
        raised ModuleNotFoundError, the prior `except ImportError: ... self._load_ip_adapter()`
        block swallowed it, and 600 images were generated with weak IP-Adapter).
        """
        try:
            from pulid.pipeline_flux import PuLIDPipeline  # type: ignore

            self._pulid = PuLIDPipeline(self._pipeline, device=self.device)
            # PuLID's load_pretrain takes pretrain_path (a local file path) as
            # its first arg, NOT a HF repo id — the repo 'guozinan/PuLID' is
            # hardcoded inside the library. Calling with no args uses defaults
            # (downloads pulid_flux_v0.9.0.safetensors to ./models/ via HF Hub).
            self._pulid.load_pretrain()
            logger.info("PuLID-Flux adapter loaded")
        except Exception as e:
            logger.error(
                f"PuLID load FAILED ({type(e).__name__}: {e}). "
                "Per CLAUDE.md item 2, refusing to silently fall back to IP-Adapter. "
                "Common causes:\n"
                "  - Missing transitive deps: pip install ftfy regex torchsde\n"
                "  - PuLID source not on sys.path: clone https://github.com/ToTheBeginning/PuLID "
                "and add it via sys.path or set the actor's pulid_src parameter\n"
                "  - PuLID model weights not yet downloaded — check HF_TOKEN access to "
                f"{self.pulid_model_id!r}\n"
                "If you genuinely want IP-Adapter, set identity_path='ip_adapter' explicitly."
            )
            raise

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

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skip-if-exists pre-pass: build the full counterfactual list, partition into
        # already-generated vs missing. If everything's already there, return without
        # touching the model (cheap reruns of partial jobs).
        axes = request.counterfactual_axes
        keys = list(axes.keys())
        all_combos = list(itertools.product(*[axes[k] for k in keys]))

        existing_results: list[GenerationResult] = []
        missing: list[tuple] = []
        for combo in all_combos:
            axis_values = dict(zip(keys, combo))
            counterfactual_id = self._make_id(request.seed_identity_id, axis_values)
            image_path = output_dir / f"{counterfactual_id}.png"
            if image_path.exists():
                existing_results.append(
                    GenerationResult(
                        seed_identity_id=request.seed_identity_id,
                        counterfactual_id=counterfactual_id,
                        image_path=str(image_path),
                        prompt_used="",  # not regenerated; manifest can recover from prior run
                        axis_values=axis_values,
                        metadata={"skipped": True, "reason": "image_exists"},
                    )
                )
            else:
                missing.append(combo)

        if not missing:
            logger.info(
                f"All {len(all_combos)} counterfactuals for {request.seed_identity_id} "
                f"already exist in {output_dir}; skipping load."
            )
            return existing_results

        if existing_results:
            logger.info(
                f"{request.seed_identity_id}: {len(existing_results)}/{len(all_combos)} "
                f"already generated; producing {len(missing)} missing."
            )

        self._lazy_load()

        seed_img = Image.open(request.seed_image_path).convert("RGB") if request.seed_image_path else None
        control_image = self._build_control_image(seed_img) if seed_img is not None else None

        id_embedding = None
        if self.identity_path != "none" and request.seed_image_path:
            id_embedding = self._extract_id_embedding(request.seed_image_path)

        results: list[GenerationResult] = list(existing_results)

        for combo in missing:
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
                # InstantX/FLUX.1-dev-Controlnet-Union requires an integer
                # control_mode telling it which conditioning type this is.
                # Without it, FluxControlNetModel.forward raises ValueError.
                pipe_kwargs["control_mode"] = _CONTROLNET_UNION_MODE_INDEX[self.controlnet_mode]
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
