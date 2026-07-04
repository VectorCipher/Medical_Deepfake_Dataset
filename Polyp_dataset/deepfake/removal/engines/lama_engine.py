"""
engines/lama_engine.py
=======================
Wraps `simple-lama-inpainting`, which auto-downloads the pretrained
big-lama checkpoint on first use (needs Internet: On in Kaggle notebook
settings the first time).

Install:  pip install simple-lama-inpainting
Source:   https://github.com/advimman/lama (Samsung AI Center / Skoltech)
"""

import cv2
import numpy as np
from PIL import Image

from config import DEVICE


class LamaEngine:
    name = "lama"

    def __init__(self):
        from simple_lama_inpainting import SimpleLama
        self.model = SimpleLama(device=DEVICE)

    def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.model(Image.fromarray(img_rgb), Image.fromarray(mask))
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
