"""Flux + PuLID + ControlNet via PuLID's native architecture.

Why this exists
---------------
The original `flux_pulid.py` builds on diffusers' `FluxControlNetPipeline` and
attempts to attach PuLID via `PuLIDPipeline(diffusers_pipeline)`. That path is
broken by design: PuLID's `__init__` does

    dit.pulid_ca = self.pulid_ca   # attach attributes to the transformer
    dit.pulid_double_interval = ...

and PuLID's *own* `flux/model.py` Flux class has a forward that reads those
attributes. **Diffusers' `FluxTransformer2DModel.forward` does not.** When you
hand PuLID a diffusers pipeline, the attributes attach but never fire — so
generation runs as plain text-to-image Flux, identity is never injected, and
ArcFace cosine sim to the seed lands near 0 (Review 001 §4 / MVP's
`preserved_fraction = 0.0`).

The fix
-------
Use PuLID's *native* Flux class (which DOES honor `pulid_ca`), but extend its
forward to also apply diffusers-format ControlNet residuals at the appropriate
block boundaries. Pose ControlNet (CLAUDE.md item 1) and PuLID identity
injection (CLAUDE.md item 2) both hold.

Architecture
------------
- Flux DiT: `FluxWithControlNet` — subclasses PuLID's `flux.model.Flux`,
  copies its forward, adds the ControlNet residual application points exactly
  where diffusers' `FluxTransformer2DModel.forward` applies them, **without
  removing** PuLID's pulid_ca insertions.
- T5 / CLIP / AE: PuLID's loaders (`flux.util.load_t5/load_clip/load_ae`),
  which produce models compatible with PuLID's Flux.
- ControlNet: diffusers' `FluxControlNetModel.from_pretrained` for the
  ControlNet-Union weights — invoked standalone (we don't go through the
  full `FluxControlNetPipeline`).
- Denoise loop: a custom loop modeled on PuLID's `flux.sampling.denoise`,
  extended to call ControlNet at each step and thread the resulting block
  samples through to the Flux forward.

Caller responsibility
---------------------
The caller (notebook / Ray actor / CLI) MUST add PuLID's source directory
to `sys.path` BEFORE importing this module, so `from flux.model import Flux`
resolves. The smoke / MVP notebooks already do this via the Workspace Volume
clone — see `notebooks/06_databricks_pulid_diagnostic.ipynb` for the canonical
pattern.
"""
from __future__ import annotations

import gc
import itertools
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from cap.generator.base import CounterfactualGenerator, GenerationRequest, GenerationResult
from cap.generator.prompts import build_demographic_prompt
from cap.utils.logging import get_logger

logger = get_logger()


# --------------------------------------------------------------------- helpers

def _import_pulid_flux():
    """Import PuLID's flux module (lazy + assert sys.path is set up).

    PuLID's `flux/` is a top-level package under the cloned repo, not a real
    pip-installed dependency. Caller must have done `sys.path.insert(0,
    pulid_src)` before reaching this module.
    """
    try:
        from flux.model import Flux as _PuLIDFlux  # type: ignore
        from flux.modules.layers import timestep_embedding  # type: ignore
        from flux.sampling import get_noise, get_schedule, prepare, unpack  # type: ignore
        from flux.util import (  # type: ignore
            load_ae,
            load_clip,
            load_flow_model,
            load_t5,
        )
        from pulid.pipeline_flux import PuLIDPipeline  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import PuLID's flux module: {e}. "
            "Caller must add PuLID source directory to sys.path before importing "
            "cap.generator.flux_pulid_native (see notebook 06 for the pattern)."
        ) from e
    return {
        "Flux": _PuLIDFlux,
        "timestep_embedding": timestep_embedding,
        "get_noise": get_noise,
        "get_schedule": get_schedule,
        "prepare": prepare,
        "unpack": unpack,
        "load_ae": load_ae,
        "load_clip": load_clip,
        "load_flow_model": load_flow_model,
        "load_t5": load_t5,
        "PuLIDPipeline": PuLIDPipeline,
    }


# ----------------------------- extended Flux model with ControlNet residuals

def _make_flux_with_controlnet_class():
    """Factory: build a Flux subclass at import time once PuLID's Flux is on sys.path.

    Returns the FluxWithControlNet class. We can't define it at module import
    because PuLID's `flux.model.Flux` is only resolvable after the caller sets
    sys.path; defining a subclass at module top-level would fail.
    """
    pulid = _import_pulid_flux()
    Flux = pulid["Flux"]
    timestep_embedding = pulid["timestep_embedding"]

    class FluxWithControlNet(Flux):
        """PuLID's Flux + ControlNet residual application.

        Forward is copied from PuLID's flux/model.py with two additions:

          1. After each double_block(i), if controlnet_block_samples is
             provided, add residual `controlnet_block_samples[i // interval]`.
             This matches diffusers' FluxControlNet residual injection.
          2. After each single_block(i), apply controlnet_single_block_samples
             with the same interval logic.

        PuLID's existing `pulid_ca` modulations remain in place at their
        original intervals (every `pulid_double_interval` / `pulid_single_interval`
        blocks) — both effects compose.
        """

        def forward(
            self,
            img: Tensor,
            img_ids: Tensor,
            txt: Tensor,
            txt_ids: Tensor,
            timesteps: Tensor,
            y: Tensor,
            guidance: Tensor = None,
            id: Tensor = None,
            id_weight: float = 1.0,
            controlnet_block_samples: list[Tensor] | None = None,
            controlnet_single_block_samples: list[Tensor] | None = None,
            aggressive_offload: bool = False,
        ) -> Tensor:
            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError("Input img and txt tensors must have 3 dimensions.")

            img = self.img_in(img)
            vec = self.time_in(timestep_embedding(timesteps, 256))
            if self.params.guidance_embed:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
            vec = vec + self.vector_in(y)
            txt = self.txt_in(txt)

            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)

            # Match diffusers' interval-based residual distribution: ControlNet
            # may have fewer layers than the Flux DiT, so each residual gets
            # applied across `interval` consecutive blocks.
            cn_double_interval = (
                int(math.ceil(len(self.double_blocks) / len(controlnet_block_samples)))
                if controlnet_block_samples
                else None
            )
            cn_single_interval = (
                int(math.ceil(len(self.single_blocks) / len(controlnet_single_block_samples)))
                if controlnet_single_block_samples
                else None
            )

            ca_idx = 0
            for i, block in enumerate(self.double_blocks):
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

                if cn_double_interval is not None:
                    j = i // cn_double_interval
                    if j < len(controlnet_block_samples):
                        img = img + controlnet_block_samples[j]

                if i % self.pulid_double_interval == 0 and id is not None:
                    img = img + id_weight * self.pulid_ca[ca_idx](id, img)
                    ca_idx += 1

            img = torch.cat((txt, img), 1)
            for i, block in enumerate(self.single_blocks):
                x = block(img, vec=vec, pe=pe)
                real_img, txt = x[:, txt.shape[1]:, ...], x[:, :txt.shape[1], ...]

                if cn_single_interval is not None:
                    j = i // cn_single_interval
                    if j < len(controlnet_single_block_samples):
                        real_img = real_img + controlnet_single_block_samples[j]

                if i % self.pulid_single_interval == 0 and id is not None:
                    real_img = real_img + id_weight * self.pulid_ca[ca_idx](id, real_img)
                    ca_idx += 1

                img = torch.cat((txt, real_img), 1)

            img = img[:, txt.shape[1]:, ...]
            img = self.final_layer(img, vec)
            return img

    return FluxWithControlNet


# ---------------------------------------------- ControlNet residuals from a control image

def _compute_controlnet_residuals(
    controlnet,
    control_image_latent: Tensor,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    vec: Tensor,
    guidance_vec: Tensor,
    controlnet_mode: int,
    controlnet_conditioning_scale: float,
):
    """Call diffusers' FluxControlNetModel and return its block samples.

    Diffusers' FluxControlNetModel matches the Flux DiT's API closely; we feed
    it the same hidden state + img/txt embeddings + control image and it
    returns the list-of-tensors residuals to add at each transformer block.
    """
    # Both PuLID's Flux and diffusers' FluxControlNetModel use the same
    # normalized timestep convention ([0, 1] from get_schedule); no scaling.
    out = controlnet(
        hidden_states=img,
        controlnet_cond=control_image_latent,
        controlnet_mode=torch.tensor([controlnet_mode], device=img.device, dtype=torch.long),
        conditioning_scale=controlnet_conditioning_scale,
        encoder_hidden_states=txt,
        pooled_projections=vec,
        timestep=timesteps,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance_vec,
        return_dict=False,
    )
    # diffusers FluxControlNetModel returns (controlnet_block_samples, controlnet_single_block_samples)
    return out


# --------------------------------------------------------- the generator class

# InstantX/FLUX.1-dev-Controlnet-Union mode index
_CONTROLNET_UNION_MODE_INDEX = {
    "canny": 0,
    "tile": 1,
    "depth": 2,
    "blur": 3,
    "pose": 4,
    "gray": 5,
    "lq": 6,
}


class FluxPuLIDNativeGenerator(CounterfactualGenerator):
    """Counterfactual generator using PuLID's native Flux + ControlNet residual injection.

    Identical request/response interface to FluxPuLIDControlNetGenerator (the
    diffusers-based one), so callers can swap by class name without any other
    code change.
    """

    def __init__(
        self,
        flux_model_name: str = "flux-dev",
        controlnet_model_id: str = "InstantX/FLUX.1-dev-Controlnet-Union",
        device: str = "cuda",
        cache_dir: str | None = None,
        controlnet_mode: str = "pose",
        face_model_name: str = "antelopev2",
        id_weight: float = 1.0,
        controlnet_conditioning_scale: float = 0.6,
        weight_dtype: torch.dtype = torch.bfloat16,
        offload: bool = False,
    ):
        self.flux_model_name = flux_model_name
        self.controlnet_model_id = controlnet_model_id
        self.device = device
        self.cache_dir = cache_dir
        self.controlnet_mode = controlnet_mode
        self.face_model_name = face_model_name
        self.id_weight = id_weight
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.weight_dtype = weight_dtype
        self.offload = offload

        # Lazily-loaded
        self._flux: Any = None
        self._ae: Any = None
        self._t5: Any = None
        self._clip: Any = None
        self._controlnet: Any = None
        self._pulid: Any = None
        self._pulid_modules: Any = None
        self._control_processor: Any = None

    # ------------------------------------------------------------- model load

    def _lazy_load(self) -> None:
        if self._flux is not None:
            return

        self._pulid_modules = _import_pulid_flux()
        FluxWithControlNet = _make_flux_with_controlnet_class()

        # ---- Flux DiT (PuLID's loader, but instantiate our subclass) ------
        # PuLID's `load_flow_model` returns a `Flux` instance with weights loaded.
        # We instantiate `FluxWithControlNet` instead and copy the state dict.
        logger.info(f"Loading Flux ({self.flux_model_name}) via PuLID's loader...")
        t0 = time.time()
        baseline = self._pulid_modules["load_flow_model"](
            self.flux_model_name,
            device="cpu" if self.offload else self.device,
        )
        # Build our subclass with the same params and copy weights.
        self._flux = FluxWithControlNet(baseline.params).to(
            "cpu" if self.offload else self.device,
            self.weight_dtype,
        )
        self._flux.load_state_dict(baseline.state_dict(), strict=True)
        self._flux.eval()
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Flux DiT loaded in {time.time() - t0:.1f}s")

        # ---- T5 / CLIP / AE ----------------------------------------------
        t0 = time.time()
        self._t5 = self._pulid_modules["load_t5"](self.device, max_length=128)
        self._clip = self._pulid_modules["load_clip"](self.device)
        self._ae = self._pulid_modules["load_ae"](
            self.flux_model_name,
            device="cpu" if self.offload else self.device,
        )
        logger.info(f"T5/CLIP/AE loaded in {time.time() - t0:.1f}s")

        # ---- ControlNet (diffusers, just for the model & forward) --------
        t0 = time.time()
        from diffusers import FluxControlNetModel

        self._controlnet = FluxControlNetModel.from_pretrained(
            self.controlnet_model_id,
            torch_dtype=self.weight_dtype,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self._controlnet.eval()
        logger.info(f"ControlNet loaded in {time.time() - t0:.1f}s")

        # ---- PuLID identity pipeline (attaches pulid_ca to our flux) -----
        t0 = time.time()
        PuLIDPipeline = self._pulid_modules["PuLIDPipeline"]
        self._pulid = PuLIDPipeline(
            self._flux,
            device=self.device,
            weight_dtype=self.weight_dtype,
        )
        self._pulid.load_pretrain()
        logger.info(f"PuLID adapter + face encoder loaded in {time.time() - t0:.1f}s")

        # ---- Structural preprocessor for the control image ---------------
        # Use Canny (cv2-only, no controlnet_aux mediapipe dep) when
        # controlnet_mode is canny; OpenposeDetector for pose.
        from cap.generator.structural import build_control_processor
        self._control_processor = build_control_processor(self.controlnet_mode, device=self.device)

    # ----------------------------------------------------- per-image helpers

    def _build_control_image(self, seed_image: Image.Image) -> Image.Image:
        return self._control_processor(seed_image)

    def _encode_control_image(self, control_image: Image.Image, height: int, width: int) -> Tensor:
        """Encode a PIL control image to the latent shape ControlNet expects."""
        from torchvision import transforms

        tx = transforms.Compose([
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        t = tx(control_image.convert("RGB")).unsqueeze(0).to(self.device, self.weight_dtype)
        # Encode to latent space via the AE (matches FluxControlNetModel's expected input).
        with torch.no_grad():
            latent = self._ae.encode(t * 2.0 - 1.0)  # [-1, 1] range
        # diffusers' FluxControlNetModel expects packed latent [B, seq, C].
        # Pack from [B, C, H/8, W/8] → [B, (H/16 * W/16), C*4].
        from einops import rearrange
        latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        return latent

    @torch.inference_mode()
    def _denoise_with_controlnet(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        id_emb: Tensor | None,
        uncond_id: Tensor | None,
        control_image_latent: Tensor | None,
    ) -> Image.Image:
        """Custom denoise loop: ControlNet residuals + PuLID id injection per step."""
        get_noise = self._pulid_modules["get_noise"]
        get_schedule = self._pulid_modules["get_schedule"]
        prepare = self._pulid_modules["prepare"]
        unpack = self._pulid_modules["unpack"]

        torch.manual_seed(seed)
        x = get_noise(1, height, width, device=self.device, dtype=self.weight_dtype, seed=seed)
        prep = prepare(t5=self._t5, clip=self._clip, img=x, prompt=prompt)
        timesteps = get_schedule(num_inference_steps, prep["img"].shape[1], shift=True)

        guidance_vec = torch.full(
            (prep["img"].shape[0],), guidance_scale, device=self.device, dtype=self.weight_dtype
        )

        img = prep["img"]
        img_ids = prep["img_ids"]
        txt = prep["txt"]
        txt_ids = prep["txt_ids"]
        vec = prep["vec"]

        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=self.weight_dtype, device=self.device)

            # ControlNet residuals for this step (only if a control image is provided)
            cn_block_samples = None
            cn_single_block_samples = None
            if control_image_latent is not None:
                out = _compute_controlnet_residuals(
                    self._controlnet,
                    control_image_latent,
                    img, img_ids, txt, txt_ids, t_vec, vec, guidance_vec,
                    controlnet_mode=_CONTROLNET_UNION_MODE_INDEX[self.controlnet_mode],
                    controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                )
                cn_block_samples, cn_single_block_samples = out

            pred = self._flux(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                id=id_emb,
                id_weight=self.id_weight,
                controlnet_block_samples=cn_block_samples,
                controlnet_single_block_samples=cn_single_block_samples,
            )

            img = img + (t_prev - t_curr) * pred

        # Decode latent → image
        img = unpack(img.float(), height, width)
        with torch.autocast(device_type=self.device, dtype=self.weight_dtype):
            img = self._ae.decode(img)
        img = (img.clamp(-1, 1) + 1) / 2
        img = (img * 255).to(torch.uint8).cpu().numpy()[0].transpose(1, 2, 0)
        return Image.fromarray(img)

    # ------------------------------------------------------- request / generate

    def generate(
        self, request: GenerationRequest, output_dir: str | Path
    ) -> list[GenerationResult]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skip-if-exists pre-pass (matches diffusers-path behavior)
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
                        prompt_used="",
                        axis_values=axis_values,
                        metadata={"skipped": True, "reason": "image_exists"},
                    )
                )
            else:
                missing.append(combo)

        if not missing:
            return existing_results

        self._lazy_load()

        # Seed image → control image latent + identity embedding
        seed_img = (
            Image.open(request.seed_image_path).convert("RGB")
            if request.seed_image_path
            else None
        )
        control_image_pil = self._build_control_image(seed_img) if seed_img is not None else None
        control_image_latent = (
            self._encode_control_image(control_image_pil, request.height, request.width)
            if control_image_pil is not None
            else None
        )

        id_emb = None
        uncond_id = None
        if request.seed_image_path:
            id_arr = np.array(seed_img)
            with torch.inference_mode():
                id_emb, uncond_id = self._pulid.get_id_embedding(id_arr, cal_uncond=False)

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
            image = self._denoise_with_controlnet(
                prompt=prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                id_emb=id_emb,
                uncond_id=uncond_id,
                control_image_latent=control_image_latent,
            )
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
                        "id_weight": self.id_weight,
                        "controlnet_mode": self.controlnet_mode,
                        "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
                        "elapsed_s": round(elapsed, 2),
                    },
                )
            )

        gc.collect()
        torch.cuda.empty_cache()
        return results

    @staticmethod
    def _make_id(seed_id: str, axis_values: dict[str, Any]) -> str:
        parts = [seed_id]
        for k in sorted(axis_values.keys()):
            parts.append(f"{k}{axis_values[k]}")
        return "_".join(str(p) for p in parts)

    def model_versions(self) -> dict[str, str]:
        return {
            "flux": self.flux_model_name,
            "controlnet": self.controlnet_model_id,
            "controlnet_mode": self.controlnet_mode,
            "id_weight": str(self.id_weight),
            "controlnet_conditioning_scale": str(self.controlnet_conditioning_scale),
            "face_model": self.face_model_name,
            "weight_dtype": str(self.weight_dtype).replace("torch.", ""),
            "architecture": "pulid_native_with_controlnet",
        }
