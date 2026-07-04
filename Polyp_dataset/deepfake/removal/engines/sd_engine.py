"""
engines/sd_engine.py
=====================
Fallback engine for cases LaMa can't clean up after a couple of tries.
Semantic fill (prompted) tends to handle larger/oddly-shaped removal
regions better than LaMa, at the cost of speed.

Install:  pip install diffusers accelerate transformers
"""

import cv2
import numpy as np
from PIL import Image

from config import DEVICE, SD_DTYPE


class SDInpaintEngine:
    name = "sd"

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-inpainting"):
        from diffusers import AutoPipelineForInpainting
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            model_id, torch_dtype=SD_DTYPE
        ).to(DEVICE)
        self.pipe.set_progress_bar_config(disable=True)

    def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        orig_size = (img_bgr.shape[1], img_bgr.shape[0])
        img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).resize((512, 512))
        mask_img = Image.fromarray(mask).resize((512, 512))

        out = self.pipe(
            prompt="smooth healthy pink intestinal mucosa, endoscopy, seamless texture, no lesion",
            negative_prompt="polyp, lesion, growth, artifact, blur, discoloration",
            image=img_rgb,
            mask_image=mask_img,
            num_inference_steps=25,
            guidance_scale=6.0,
        ).images[0].resize(orig_size)

        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
