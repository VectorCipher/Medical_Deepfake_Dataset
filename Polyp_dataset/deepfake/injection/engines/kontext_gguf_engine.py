"""
engines/kontext_gguf_engine.py
================================
FLUX.1 Kontext [dev] engine, loaded from a GGUF-quantized transformer via
diffusers, for reference-guided lesion INJECTION.

Memory strategy (Q8_0 on 16 GB T4):
  The Q8_0 transformer alone is ~12 GB.  enable_sequential_cpu_offload()
  is incompatible with GGUF tensors, so we use enable_model_cpu_offload()
  which moves whole sub-models on/off GPU.  To stay within VRAM we also:
    1. Generate at a small resolution (max 256px) so the VAE tiles fit.
    2. Shrink VAE tile sizes to 256 pixels.
    3. Cap the reference crop at 128px.
    4. Aggressively gc.collect + empty_cache before each call.
  The result is upscaled back to the original image size before saving.

Requires:
    pip install git+https://github.com/huggingface/diffusers.git
    pip install transformers accelerate sentencepiece gguf

License note: FLUX.1 Kontext [dev] weights are under BFL's non-commercial
research license - fine for building a research dataset, flag before any
commercial reuse.
"""

import gc
import os
import numpy as np
import cv2
from PIL import Image
import torch

from config import (
    DEVICE, KONTEXT_GGUF_PATH, KONTEXT_BASE_REPO,
    KONTEXT_NUM_STEPS, KONTEXT_GUIDANCE_SCALE, KONTEXT_STRENGTH,
)


class KontextGGUFEngine:
    name = "kontext_gguf"

    def __init__(
        self,
        gguf_path: str = None,
        base_repo: str = None,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        strength: float = None,
    ):
        from diffusers import (
            FluxKontextInpaintPipeline,
            FluxTransformer2DModel,
            GGUFQuantizationConfig,
        )
        from huggingface_hub import hf_hub_download

        gguf_path = gguf_path or KONTEXT_GGUF_PATH
        base_repo = base_repo or KONTEXT_BASE_REPO

        # Download transformer config from the correct subfolder
        config_path = hf_hub_download(
            repo_id=base_repo, filename="config.json", subfolder="transformer"
        )
        config_dir = os.path.dirname(config_path)

        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            config=config_dir,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )

        self.pipe = FluxKontextInpaintPipeline.from_pretrained(
            base_repo,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        # enable_model_cpu_offload is the ONLY offload mode compatible with
        # GGUF quantized tensors (sequential offload crashes on meta-device
        # conversion for GGUF types).
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        # Shrink VAE tile size to reduce per-tile VRAM
        self.pipe.vae.tile_sample_min_size = 256
        self.pipe.vae.tile_latent_min_size = 32
        self.pipe.set_progress_bar_config(disable=True)

        self.num_inference_steps = num_inference_steps or KONTEXT_NUM_STEPS
        self.guidance_scale = guidance_scale or KONTEXT_GUIDANCE_SCALE
        self.strength = strength or KONTEXT_STRENGTH

    def inject(
        self,
        target_img_bgr: np.ndarray,
        reference_crop_bgr: np.ndarray,
        mask: np.ndarray,
        prompt: str = (
            "a realistic colorectal polyp lesion, matching the surrounding "
            "mucosa lighting, texture and color, seamlessly integrated"
        ),
        seed: int = None,
    ) -> np.ndarray:
        orig_size = (target_img_bgr.shape[1], target_img_bgr.shape[0])

        # ----- Downscale to 256px max to fit within ~2 GB VRAM headroom -----
        # Q8_0 transformer eats ~12 GB; on a 16 GB T4 this leaves very
        # little for VAE encode/decode even with tiling.  256px generation
        # uses ~4x less memory than 512px per tile.
        max_dim = 256
        scale = min(1.0, max_dim / max(orig_size))
        new_w = max(16, (int(orig_size[0] * scale) // 16) * 16)
        new_h = max(16, (int(orig_size[1] * scale) // 16) * 16)

        target_img_bgr = cv2.resize(
            target_img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

        # Cap reference crop at 128px to keep memory bounded
        ref_h, ref_w = reference_crop_bgr.shape[:2]
        if max(ref_h, ref_w) > 128:
            ref_scale = 128 / max(ref_h, ref_w)
            reference_crop_bgr = cv2.resize(
                reference_crop_bgr,
                (int(ref_w * ref_scale), int(ref_h * ref_scale)),
                interpolation=cv2.INTER_AREA,
            )

        target_rgb = Image.fromarray(cv2.cvtColor(target_img_bgr, cv2.COLOR_BGR2RGB))
        ref_rgb = Image.fromarray(cv2.cvtColor(reference_crop_bgr, cv2.COLOR_BGR2RGB))
        mask_img = Image.fromarray(mask)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Force free any stale GPU tensors before inference
        gc.collect()
        torch.cuda.empty_cache()

        result = self.pipe(
            prompt=prompt,
            image=target_rgb,
            mask_image=mask_img,
            image_reference=ref_rgb,
            height=new_h,
            width=new_w,
            strength=self.strength,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]

        # Free GPU memory immediately after generation
        gc.collect()
        torch.cuda.empty_cache()

        result = result.resize(orig_size)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
