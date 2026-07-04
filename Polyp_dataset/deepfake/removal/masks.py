"""
masks.py
========
Loads your real binary masks (preferred) and falls back to bbox-derived
rectangles only if a mask file is missing for a given image. Also handles
mask dilation, which is how the refinement loop "tries harder" on a
failed iteration.
"""

import os
import cv2
import numpy as np

from config import MASKS_DIR, MASK_FILE_EXT, MASK_DILATE_INIT


def load_mask(filename, img_shape):
    """
    Loads the real binary mask for `filename`. Returns a uint8 mask
    (0/255) resized to img_shape if needed. Returns None if not found,
    so the caller can fall back to a bbox rectangle.
    """
    base = os.path.splitext(filename)[0]
    mask_path = os.path.join(MASKS_DIR, base + MASK_FILE_EXT)
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    h, w = img_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # binarize defensively in case the mask has anti-aliased edges or is 0/1 instead of 0/255
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def bbox_fallback_mask(img_shape, bboxes, pad=8):
    """
    Creates a mask from one or more bounding boxes.
    bboxes: list of (x_min, y_min, x_max, y_max) tuples.
    Used only when a real mask file is missing for an image.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
        mask[y1p:y2p, x1p:x2p] = 255
    return mask


def prepare_initial_mask(filename, img, bboxes=None):
    """
    Returns the mask to use for the first inpainting attempt: the real
    mask if available (lightly dilated to avoid boundary halo), else a
    combined bbox rectangle from the per-image CSV.
    """
    mask = load_mask(filename, img.shape)
    if mask is None:
        if bboxes is None or len(bboxes) == 0:
            raise FileNotFoundError(
                f"No mask found for {filename} and no bbox fallback provided."
            )
        mask = bbox_fallback_mask(img.shape, bboxes)
        return mask, "bbox_fallback"

    mask = dilate_mask(mask, MASK_DILATE_INIT)
    return mask, "real_mask"


def dilate_mask(mask, px):
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    return cv2.dilate(mask, kernel, iterations=1)

