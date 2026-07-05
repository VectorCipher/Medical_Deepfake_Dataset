"""
exemplar_bank.py
=================
Builds a library of real polyp/tumor crops from your EXISTING dataset
(the one with real images + real masks, used earlier for the removal
task). These crops become the `reference_crop_bgr` fed into
KontextGGUFEngine.inject() - this is what lets injection use a real
lesion appearance instead of a text-only hallucination.

Run once before running main_inject.py.
"""

import os
import cv2
import numpy as np
import pandas as pd

from config import IMAGES_DIR, MASKS_DIR, MASK_FILE_EXT, EXEMPLAR_BANK_DIR


def build_exemplar_bank(padding: int = 6):
    """
    For every image with a real mask, crops the tight bounding box of the
    mask (plus a small margin) and saves it as its own small image, along
    with an alpha-style companion mask so injection can respect the
    lesion's real silhouette rather than a plain rectangle.
    """
    os.makedirs(EXEMPLAR_BANK_DIR, exist_ok=True)

    manifest = []
    for fname in os.listdir(IMAGES_DIR):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(IMAGES_DIR, fname)
        mask_path = os.path.join(MASKS_DIR, os.path.splitext(fname)[0] + MASK_FILE_EXT)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        h, w = img.shape[:2]
        x1, x2 = max(0, xs.min() - padding), min(w, xs.max() + padding)
        y1, y2 = max(0, ys.min() - padding), min(h, ys.max() + padding)

        crop = img[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        base = os.path.splitext(fname)[0]
        crop_path = os.path.join(EXEMPLAR_BANK_DIR, f"{base}_crop.png")
        crop_mask_path = os.path.join(EXEMPLAR_BANK_DIR, f"{base}_crop_mask.png")
        cv2.imwrite(crop_path, crop)
        cv2.imwrite(crop_mask_path, crop_mask)

        manifest.append({
            "source_image": fname,
            "crop_path": crop_path,
            "crop_mask_path": crop_mask_path,
            "crop_w": x2 - x1,
            "crop_h": y2 - y1,
        })

    manifest_path = os.path.join(EXEMPLAR_BANK_DIR, "manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"Built {len(manifest)} exemplars -> {EXEMPLAR_BANK_DIR}")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def load_exemplar_bank():
    """Returns the manifest DataFrame; use this in main_inject.py to sample from."""
    manifest_path = os.path.join(EXEMPLAR_BANK_DIR, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No exemplar manifest at {manifest_path} - run build_exemplar_bank() first."
        )
    return pd.read_csv(manifest_path)


if __name__ == "__main__":
    build_exemplar_bank()
