"""
pipeline.py
===========
The self-refining loop, decoupled from any specific engine or critic
implementation so you can swap pieces (e.g. add a new engine) without
touching this file.
"""

import os
import cv2
from dataclasses import dataclass

from config import (
    IMAGES_DIR, OUTPUT_DIR, MAX_ITERS, CONF_THRESHOLD,
    SWITCH_TO_SD_AFTER, MASK_DILATE_STEP,
)
from masks import prepare_initial_mask, dilate_mask


@dataclass
class ImageResult:
    filename: str
    iterations_used: int
    final_confidence: float
    engine_used: str
    mask_source: str
    success: bool


def refine_and_inpaint(filename, engines: dict, critic, bbox=None):
    """
    engines: dict like {"lama": LamaEngine_instance, "sd": SDInpaintEngine_instance}
    bbox: optional (x1, y1, x2, y2), used only if a real mask file is missing
    """
    img_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        return ImageResult(filename, 0, 1.0, "none", "none", False)

    mask, mask_source = prepare_initial_mask(filename, img, bbox=bbox)

    engine_name = "lama"
    result_img = img.copy()
    conf = 1.0

    for it in range(1, MAX_ITERS + 1):
        engine = engines[engine_name]
        result_img = engine.inpaint(img, mask)

        conf = critic.max_confidence(result_img)
        if conf < CONF_THRESHOLD:
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), result_img)
            return ImageResult(filename, it, conf, engine_name, mask_source, True)

        # escalate for next attempt
        mask = dilate_mask(mask, MASK_DILATE_STEP)
        if it >= SWITCH_TO_SD_AFTER and engine_name == "lama" and "sd" in engines:
            engine_name = "sd"

    # exhausted iterations - save best effort, flagged in filename + log
    cv2.imwrite(os.path.join(OUTPUT_DIR, "unresolved_" + filename), result_img)
    return ImageResult(filename, MAX_ITERS, conf, engine_name, mask_source, False)
