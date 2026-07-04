"""
critic.py
=========
The critic judges whether an inpainted image still "looks like" it has
a polyp. This is what makes the loop self-refining rather than one-shot.

It's a YOLOv8 detector - either your own, trained on the bboxes you
already have (see train_critic.py), or any pretrained polyp detector
checkpoint you point CRITIC_WEIGHTS at.
"""

import numpy as np
from config import DEVICE


class Critic:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)

    def max_confidence(self, img_bgr: np.ndarray) -> float:
        """
        Highest polyp-class confidence found anywhere in the image.
        Returns 0.0 if nothing is detected.
        """
        results = self.model.predict(img_bgr, verbose=False, device=DEVICE)[0]
        if len(results.boxes) == 0:
            return 0.0
        return float(results.boxes.conf.cpu().numpy().max())
