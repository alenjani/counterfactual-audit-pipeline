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
            configs as flux_configs,
            load_ae,
            load_clip,
            load_flow_model,
            load_flow_model_quintized,
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
        "configs": flux_configs,
        "timestep_embedding": timestep_embedding,
        "get_noise": get_noise,
        "get_schedule": get_schedule,
        "prepare": prepare,
        "unpack": unpack,
        "load_ae": load_ae,
        "load_clip": load_clip,
        "load_flow_model": load_flow_model,
        "load_flow_model_quintized": load_flow_model_quintized,
        "load_t5": load_t5,
        "PuLIDPipeline": PuLIDPipeline,
    }


def _load_flux_quintized_low_peak(pulid_modules, name: str, device):
    """Drop-in replacement for PuLID's `load_flow_model_quintized` that avoids
    BOTH the 36 GB CPU peak (PuLID stock) AND the 24 GB GPU peak (optimum-
    quanto stock requantize) when targeting a 22 GB L4.

    The two stock paths' failure modes:
      - PuLID's `load_flow_model_quintized`: builds a BF16 Flux skeleton on
        CPU (~24 GB) then loads the FP8 state dict (~12 GB) on top → peak
        ~36 GB CPU. Exceeds the ~27 GB usable on the g2-standard-16 driver
        after Spark/JVM overhead → kernel OOM during init.
      - optimum-quanto's `requantize`: materializes the BF16 model on CPU,
        moves it to `device` (24 GB on the 22 GB GPU → CUDA OOM), then loads
        the FP8 state dict (would have shrunk it to 12 GB, but never reached).

    Our path:
      1. Build Flux skeleton on `meta` device (0 GB).
      2. Replace Linear → QLinear via quanto's `_quantize_submodule` (still
         meta, 0 GB).
      3. Load FP8 state dict on CPU (~12 GB).
      4. `load_state_dict(sd, assign=True)` swaps meta params with FP8
         tensors directly — no BF16 intermediate.
      5. `model.to(device)` moves the now-FP8 model to GPU (12 GB, fits).

    Peak CPU: ~12 GB (state dict). Peak GPU: ~12 GB (final FP8 model).
    """
    import json
    import os
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from optimum.quanto.quantize import _quantize_submodule
    from safetensors.torch import load_file as load_sft

    Flux = pulid_modules["Flux"]
    configs = pulid_modules["configs"]

    ckpt_path = "models/flux-dev-fp8.safetensors"
    if not os.path.exists(ckpt_path):
        ckpt_path = hf_hub_download(
            "XLabs-AI/flux-dev-fp8", "flux-dev-fp8.safetensors"
        )
    json_path = hf_hub_download(
        "XLabs-AI/flux-dev-fp8", "flux_dev_quantization_map.json"
    )

    logger.info("Building Flux skeleton on meta device (0 GB)...")
    with init_empty_weights():
        model = Flux(configs[name].params)

    with open(json_path) as f:
        quantization_map = json.load(f)

    # Replace Linear → QLinear in-place (still meta, no memory).
    for mod_name, m in model.named_modules():
        qconfig = quantization_map.get(mod_name)
        if qconfig is not None:
            weights = None if qconfig["weights"] == "none" else qconfig["weights"]
            activations = None if qconfig["activations"] == "none" else qconfig["activations"]
            _quantize_submodule(model, mod_name, m, weights=weights, activations=activations)
    logger.info("QLinear modules installed (meta).")

    # Disable quanto's Marlin-FP8 path entirely. Two reasons:
    # 1. Memory: Marlin's per-layer packing peaks ~3× weight on GPU →
    #    cumulative load OOMs the L4.
    # 2. JIT-compile: Marlin needs a CUDA extension that quanto JIT-builds
    #    inside its package dir (`<site-packages>/optimum/quanto/library/
    #    extensions/cuda/build/`). On Databricks Ray workers that path is on
    #    a read-only ephemeral_nfs filesystem → JIT lock acquire fails.
    # Patch `.optimize` (used during load_state_dict) AND `.create` (used
    # during `.to(device)` dispatch) to skip the Marlin branch and return
    # plain WeightQBytesTensor instances. ~25-30% slower matmul, no JIT,
    # no extra GPU pressure.
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor as _WQB
    _WQB.optimize = lambda self: self  # type: ignore[method-assign]
    _orig_create = _WQB.create

    def _create_no_marlin(qtype, axis, size, stride, data, scale,
                          activation_qtype=None, requires_grad=False):
        # Plain WeightQBytesTensor — skip the in_features/out_features check
        # that would route to MarlinF8QBytesTensor on sm_89+ GPUs.
        return _WQB(qtype, axis, size, stride, data, scale,
                    activation_qtype, requires_grad)

    _WQB.create = staticmethod(_create_no_marlin)  # type: ignore[method-assign]
    logger.info("Marlin-FP8 path disabled (memory + read-only-fs friendly).")

    # Load FP8 weights DIRECTLY to the target device. safetensors can mmap-
    # load each tensor straight to CUDA, so the 12 GB state dict never lands
    # in host RAM. Critical because:
    #   - The g2-standard-16 driver has ~22 GB of usable Python heap after
    #     JVM/Spark overhead. Holding 12 GB sd on CPU + transient 12 GB GPU
    #     copy during .to(device) peaks at ~24 GB on the driver → kernel hang
    #     ("DRIVER_NOT_RESPONDING" with no Python traceback).
    # Loading straight to GPU pins peak host usage at ~0 GB (transient per-
    # tensor read buffers only) and peak GPU at 12 GB final.
    target_device = str(device) if not isinstance(device, str) else device
    logger.info(f"Loading FP8 state dict from {ckpt_path} directly to {target_device}...")
    sd = load_sft(ckpt_path, device=target_device)

    # assign=True replaces meta params with the FP8 tensors. After this,
    # model's params live on `device` — no separate .to(device) needed.
    logger.info("Assigning FP8 weights to meta params (no BF16, no CPU intermediate)...")
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        logger.warning(f"  missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        logger.warning(f"  unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    del sd
    gc.collect()
    return model


# --------------------------- forward extension via monkey-patch (no subclass)
#
# Earlier draft built a `FluxWithControlNet(Flux)` subclass and copied weights
# from a PuLID-loaded baseline into a fresh subclass instance. That doubles
# peak VRAM during construction (~48 GB) and OOMs the L4. Instead we monkey-
# patch the forward method on the *already-loaded* Flux instance — the model
# keeps its weights in place, only its forward is replaced. Peak VRAM stays
# at one model's worth (~24 GB BF16, less for quantized).

def _patch_flux_forward_for_controlnet(flux_module, timestep_embedding):
    """Replace `flux_module.forward` with a closure that applies BOTH PuLID's
    pulid_ca modulations AND diffusers-format ControlNet block residuals.

    The closure mirrors PuLID's flux/model.py forward exactly for the pulid_ca
    insertions (every `pulid_double_interval` / `pulid_single_interval` blocks)
    and adds ControlNet residual application after each block, matching
    diffusers' FluxControlNetPipeline interval distribution.
    """
    import types

    def extended_forward(
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

    flux_module.forward = types.MethodType(extended_forward, flux_module)
    return flux_module


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
        # FluxControlNetPipeline reshapes control_mode to (B, 1) before passing
        # to FluxControlNetModel — the embedder expects 2D input so its output
        # is 3D and can be concatenated to encoder_hidden_states. We mirror it.
        controlnet_mode=torch.tensor([[controlnet_mode]], device=img.device, dtype=torch.long),
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
        use_fp8: bool = True,
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
        # FP8 quantization is required to fit Flux + ControlNet + T5 on a 24 GB L4.
        # PuLID's `load_flow_model_quintized` produces FP8 weights (~12 GB) instead
        # of BF16 (~24 GB). Disable only if running on an A100/H100 with headroom.
        self.use_fp8 = use_fp8

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

        # ---- Flux DiT (PuLID's loader, then monkey-patch its forward) -----
        # Use PuLID's `load_flow_model` to get a Flux instance with weights
        # already loaded; replace its forward in-place to honor ControlNet
        # residuals while preserving PuLID's pulid_ca insertions.
        # Avoids the double-instantiation that OOMs on a 24 GB L4 BF16.
        # FP8 path uses our low-peak loader (~12 GB CPU peak) instead of
        # PuLID's stock `load_flow_model_quintized` (~36 GB peak — OOMs the
        # 64 GB driver). The low-peak path materializes weights directly on
        # `device`, so we pass self.device.
        # BF16 path (use_fp8=False) keeps PuLID's stock loader on CPU, then
        # moves to GPU. BF16 is 24 GB → won't fit on a 22 GB L4 alone; only
        # use this on a larger GPU.
        t0 = time.time()
        if self.use_fp8:
            target_device = "cpu" if self.offload else self.device
            logger.info(
                f"Loading Flux ({self.flux_model_name}, use_fp8=True) via "
                f"low-peak loader → {target_device}..."
            )
            self._flux = _load_flux_quintized_low_peak(
                self._pulid_modules, self.flux_model_name, target_device
            )
        else:
            logger.info(
                f"Loading Flux ({self.flux_model_name}, use_fp8=False) "
                f"via PuLID's load_flow_model on CPU..."
            )
            self._flux = self._pulid_modules["load_flow_model"](
                self.flux_model_name,
                device="cpu",
            )
        _patch_flux_forward_for_controlnet(
            self._flux, self._pulid_modules["timestep_embedding"]
        )
        self._flux.eval()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Flux loaded + forward patched in {time.time() - t0:.1f}s")

        # BF16 path: now move to GPU (FP8 path already on target_device).
        if not self.use_fp8 and not self.offload:
            t0 = time.time()
            self._flux = self._flux.to(self.device)
            logger.info(f"Flux moved to {self.device} in {time.time() - t0:.1f}s")
            gc.collect()
            torch.cuda.empty_cache()

        # ---- T5 / CLIP / AE on CPU ---------------------------------------
        # On a 22 GB-usable L4, simultaneously holding Flux (~12 GB FP8) +
        # ControlNet (~6 GB BF16) + T5 (~5 GB BF16) overflows. T5 is only
        # called once per generation (encode the prompt), and CLIP/AE are
        # also one-shot per image. Keeping them on CPU saves ~5 GB of GPU
        # while costing ~1-2s per generation for T5's CPU forward — fine.
        t0 = time.time()
        self._t5 = self._pulid_modules["load_t5"]("cpu", max_length=128)
        gc.collect()
        self._clip = self._pulid_modules["load_clip"]("cpu")
        gc.collect()
        self._ae = self._pulid_modules["load_ae"](
            self.flux_model_name,
            device="cpu",
        )
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"T5/CLIP/AE loaded on CPU in {time.time() - t0:.1f}s")

        # ---- ControlNet (diffusers, FP8-quantized to fit on L4) ----------
        # Layout when use_fp8=True on a 22 GB-usable L4:
        #   Flux FP8 (12 GB) + ControlNet BF16 (6 GB) + PuLID adapter (3 GB)
        #   = 21 GB resident, leaves <1 GB for activations → OOM.
        # FP8-quantizing ControlNet drops it to ~3 GB and gives ~4 GB for
        # activations. Quantize on CPU first (BF16+FP8 peak in host RAM, not
        # GPU), then move FP8 to GPU.
        t0 = time.time()
        from diffusers import FluxControlNetModel

        cn = FluxControlNetModel.from_pretrained(
            self.controlnet_model_id,
            torch_dtype=self.weight_dtype,
            cache_dir=self.cache_dir,
        )
        cn.eval()
        if self.use_fp8:
            from optimum.quanto import freeze, qfloat8, quantize as quanto_quantize

            logger.info("Quantizing ControlNet to FP8 on CPU...")
            quanto_quantize(cn, weights=qfloat8)
            freeze(cn)
            gc.collect()
        self._controlnet = cn.to(self.device)
        del cn
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"ControlNet loaded ({'FP8' if self.use_fp8 else 'BF16'}) in {time.time() - t0:.1f}s")

        # ---- PuLID identity pipeline (attaches pulid_ca to our flux) -----
        t0 = time.time()
        PuLIDPipeline = self._pulid_modules["PuLIDPipeline"]
        self._pulid = PuLIDPipeline(
            self._flux,
            device=self.device,
            weight_dtype=self.weight_dtype,
        )
        self._pulid.load_pretrain()
        gc.collect()
        torch.cuda.empty_cache()
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
        """Encode a PIL control image to the latent shape ControlNet expects.

        AE lives on CPU (memory pressure on the L4); we encode there and move
        the small latent to GPU.
        """
        from torchvision import transforms

        tx = transforms.Compose([
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        t = tx(control_image.convert("RGB")).unsqueeze(0)  # CPU, float32 [-1, 1] after scale
        with torch.no_grad():
            latent = self._ae.encode(t * 2.0 - 1.0)  # AE on CPU
        # Pack from [B, C, H/8, W/8] → [B, (H/16 * W/16), C*4] and move to GPU
        from einops import rearrange
        latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        return latent.to(self.device, self.weight_dtype)

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
        # Noise on CPU initially (matches T5/CLIP); prepare runs on CPU; we
        # move the prepared tensors to GPU before the denoise loop.
        x = get_noise(1, height, width, device="cpu", dtype=self.weight_dtype, seed=seed)
        prep = prepare(t5=self._t5, clip=self._clip, img=x, prompt=prompt)
        timesteps = get_schedule(num_inference_steps, prep["img"].shape[1], shift=True)

        # Move prepared tensors to GPU for the inference loop
        img = prep["img"].to(self.device, self.weight_dtype)
        img_ids = prep["img_ids"].to(self.device, self.weight_dtype)
        txt = prep["txt"].to(self.device, self.weight_dtype)
        txt_ids = prep["txt_ids"].to(self.device, self.weight_dtype)
        vec = prep["vec"].to(self.device, self.weight_dtype)

        guidance_vec = torch.full(
            (img.shape[0],), guidance_scale, device=self.device, dtype=self.weight_dtype
        )

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

        # Decode latent → image. AE is on CPU; move final latent there.
        img = unpack(img.float(), height, width).cpu()
        with torch.no_grad():
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
            "use_fp8": str(self.use_fp8),
            "architecture": "pulid_native_with_controlnet",
        }
