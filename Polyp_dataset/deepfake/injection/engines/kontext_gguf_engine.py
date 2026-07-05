"""
engines/kontext_gguf_engine.py
================================
FLUX.1 Kontext [dev] engine, loaded from a GGUF-quantized transformer via
diffusers, for reference-guided lesion INJECTION.

With Q4_K_M (~7 GB), enable_model_cpu_offload() leaves ~7 GB headroom
which comfortably handles 512px generation + VAE tiling.

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
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
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

        # Scale to max 512px for inference, then upscale result back
        max_dim = 512
        scale = min(1.0, max_dim / max(orig_size))
        new_w = max(16, (int(orig_size[0] * scale) // 16) * 16)
        new_h = max(16, (int(orig_size[1] * scale) // 16) * 16)

        target_img_bgr = cv2.resize(
            target_img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

        # Cap reference crop at 256px
        ref_h, ref_w = reference_crop_bgr.shape[:2]
        if max(ref_h, ref_w) > 256:
            ref_scale = 256 / max(ref_h, ref_w)
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

        gc.collect()
        torch.cuda.empty_cache()

        # Diffusers FLUX pipelines aggressively rescale image dimensions to ~1 megapixel
        # which causes OOM on 16GB GPUs. We monkey-patch the internal resolution 
        # calculator to force it to respect our requested downscaled dimensions.
        def _enforce_size(*args, **kwargs):
            return (new_h, new_w)
            
        if hasattr(self.pipe, "_default_height_width"):
            self.pipe._default_height_width = _enforce_size
        if hasattr(self.pipe, "default_height_width"):
            self.pipe.default_height_width = _enforce_size

        print(f"[DEBUG] orig_size: {orig_size}, scale: {scale:.3f}")
        print(f"[DEBUG] Resized to new_w: {new_w}, new_h: {new_h}")
        print(f"[DEBUG] target_rgb.size: {target_rgb.size}")
        print(f"[DEBUG] ref_rgb.size: {ref_rgb.size}")
        print(f"[DEBUG] mask_img.size: {mask_img.size}")
        print(f"[DEBUG] VRAM allocated before pipe: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[DEBUG] VRAM reserved before pipe: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

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

        gc.collect()
        torch.cuda.empty_cache()

        result = result.resize(orig_size)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
