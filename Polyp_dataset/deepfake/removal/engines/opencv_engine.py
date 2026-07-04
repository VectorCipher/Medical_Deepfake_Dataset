"""
engines/opencv_engine.py
==========================
Classical inpainting (Telea or Navier-Stokes). No install needed beyond
opencv-python. Not competitive with LaMa/SD on realism, but useful as:
  - a near-instant sanity check while building the pipeline
  - an "easy" difficulty tier if you want your deepfake dataset to span
    a range of fake qualities rather than only convincing ones
"""

import cv2
import numpy as np


class OpenCVEngine:
    name = "opencv"

    def __init__(self, method: str = "telea", radius: int = 5):
        self.flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        self.radius = radius

    def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(img_bgr, mask, self.radius, self.flag)
