"""
engines/kontext_gguf_engine.py
================================
FLUX.1 Kontext [dev] engine, loaded from a GGUF-quantized transformer via
diffusers, for reference-guided lesion INJECTION. Unlike the LaMa/SD
engines (which only take image + mask + text), this one is also
conditioned on a real exemplar image (a cropped real polyp/tumor), so it
blends an actual lesion appearance into the target tissue rather than
hallucinating one from text alone.

Requires (diffusers inpaint-with-reference support for Kontext is recent,
install from main branch):
    pip install git+https://github.com/huggingface/diffusers.git
    pip install transformers accelerate sentencepiece gguf

Weights:
    - Transformer (GGUF, quantized): a .gguf file from
      https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF
      (or your Kaggle mirror) - path set via config.KONTEXT_GGUF_PATH.
      Pick Q4_K_M/Q5_K_M/Q6_K depending on free VRAM.
    - VAE + text encoders + scheduler config: loaded normally from
      black-forest-labs/FLUX.1-Kontext-dev (config.KONTEXT_BASE_REPO) -
      these are NOT included in the GGUF file, which only replaces the
      transformer weights.

Known limitation worth being aware of (documented by the diffusers team
and community testers): Kontext was not originally trained with explicit
mask-based inpainting in mind - FluxKontextInpaintPipeline adds this on
top, and results can be inconsistent about respecting mask boundaries
exactly, especially for precise object placement. In practice this means:
  - it works best when the mask region is reasonably generous, not a
    razor-thin outline
  - always sanity-check a sample batch visually before running all 1000
    images through it
  - the critic-based refinement loop (injection_pipeline.py) exists
    partly to catch cases where Kontext ignores the mask and places the
    lesion somewhere the critic doesn't like, or not at all

License note: FLUX.1 Kontext [dev] weights are under BFL's non-commercial
research license - fine for building a research dataset, flag before any
commercial reuse.
"""

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

        gguf_path = gguf_path or KONTEXT_GGUF_PATH
        base_repo = base_repo or KONTEXT_BASE_REPO

        # Load only the transformer from the GGUF quant; everything else
        # (VAE, CLIP-L, T5-XXL, scheduler) comes from the original repo.
        from huggingface_hub import hf_hub_download
        import os
        
        # Download the specific config.json from the transformer subfolder
        config_path = hf_hub_download(repo_id=base_repo, filename="config.json", subfolder="transformer")
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
        # Model offloading is important here: with the critic (YOLO) also
        # resident on GPU, a 16GB T4/P100 will not fit everything at once
        # otherwise, even with a Q4 quant.
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
        """
        target_img_bgr:     the normal / ulcerative-colitis image to inject into
        reference_crop_bgr: a real polyp/tumor crop (exemplar) - see
                             exemplar_bank.py for how these are built from
                             your existing removal-task dataset + masks
        mask:                binary mask (uint8, 0/255) marking WHERE on
                              the target image to inject
        seed:                optional int for reproducibility per image
        """
        orig_size = (target_img_bgr.shape[1], target_img_bgr.shape[0])

        # VAE OOM Protection: Scale down images to max 512x512 for inference
        # (It will be scaled back up to orig_size before returning)
        max_dim = 512
        if max(orig_size) > max_dim:
            scale = max_dim / max(orig_size)
            new_w, new_h = int(orig_size[0] * scale), int(orig_size[1] * scale)
            target_img_bgr = cv2.resize(target_img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        target_rgb = Image.fromarray(cv2.cvtColor(target_img_bgr, cv2.COLOR_BGR2RGB))
        ref_rgb = Image.fromarray(cv2.cvtColor(reference_crop_bgr, cv2.COLOR_BGR2RGB))
        mask_img = Image.fromarray(mask)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=target_rgb,
            mask_image=mask_img,
            image_reference=ref_rgb,
            strength=self.strength,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]

        result = result.resize(orig_size)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
