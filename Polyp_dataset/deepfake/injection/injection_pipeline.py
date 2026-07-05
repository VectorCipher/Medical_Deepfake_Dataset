"""
injection_pipeline.py
=======================
Mirrors pipeline.py's refine-and-retry structure, but inverted: here the
critic needs to detect a lesion (not fail to detect one) for an attempt
to count as a success. Draws exemplar shape + reference crop from the
bank built by exemplar_bank.py so placement mask sizes/shapes reflect
real lesion statistics rather than arbitrary blobs.
"""

import os
import random
import cv2
import numpy as np
from dataclasses import dataclass

from config import (
    INJECTED_OUTPUT_DIR,
    INJECTION_MAX_ITERS, INJECTION_CONF_THRESHOLD,
)


@dataclass
class InjectionResult:
    filename: str
    exemplar_used: str
    iterations_used: int
    final_confidence: float
    placement_xy: tuple
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    success: bool


def sample_placement_mask(target_shape, exemplar_mask, margin_frac=0.1):
    """
    Places the exemplar's real lesion silhouette at a random valid
    location on the target image (avoiding the outer margin, where
    endoscope vignette / instrument shadows tend to sit and where a
    lesion would look anatomically implausible).
    """
    th, tw = target_shape[:2]
    mh, mw = exemplar_mask.shape[:2]

    margin_x, margin_y = int(tw * margin_frac), int(th * margin_frac)
    max_x = tw - mw - margin_x
    max_y = th - mh - margin_y
    if max_x <= margin_x or max_y <= margin_y:
        # exemplar too big for this target at 1:1 scale - downscale exemplar
        scale = 0.6
        mw, mh = int(mw * scale), int(mh * scale)
        exemplar_mask = cv2.resize(exemplar_mask, (mw, mh))
        max_x = tw - mw - margin_x
        max_y = th - mh - margin_y

    x = random.randint(margin_x, max(margin_x, max_x))
    y = random.randint(margin_y, max(margin_y, max_y))

    full_mask = np.zeros((th, tw), np.uint8)
    full_mask[y:y + mh, x:x + mw] = exemplar_mask
    return full_mask, (x, y), exemplar_mask.shape[:2]


def inject_and_refine(target_dir, filename, engine, critic, exemplar_manifest):
    img_path = os.path.join(target_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        return InjectionResult(filename, "none", 0, 0.0, (0, 0), 0, 0, 0, 0, False)

    # Create a masks subdirectory alongside injected images
    masks_out_dir = os.path.join(INJECTED_OUTPUT_DIR, "masks")
    os.makedirs(masks_out_dir, exist_ok=True)

    tried_exemplars = set()
    result_img = img.copy()
    conf = 0.0
    placement = (0, 0)
    bbox = (0, 0, 0, 0)
    placement_mask = None
    exemplar_row = None

    for it in range(1, INJECTION_MAX_ITERS + 1):
        # sample a fresh exemplar each iteration (varies both the lesion
        # appearance and the random seed, giving the loop a real chance
        # to find a combination the critic accepts)
        available = exemplar_manifest[~exemplar_manifest["crop_path"].isin(tried_exemplars)]
        pool = available if len(available) > 0 else exemplar_manifest
        exemplar_row = pool.sample(1).iloc[0]
        tried_exemplars.add(exemplar_row["crop_path"])

        ref_crop = cv2.imread(exemplar_row["crop_path"])
        ref_mask = cv2.imread(exemplar_row["crop_mask_path"], cv2.IMREAD_GRAYSCALE)

        placement_mask, placement, mask_size = sample_placement_mask(img.shape, ref_mask)
        mh, mw = mask_size
        bbox = (placement[0], placement[1], placement[0] + mw, placement[1] + mh)

        result_img = engine.inject(
            target_img_bgr=img,
            reference_crop_bgr=ref_crop,
            mask=placement_mask,
            seed=it * 1000 + hash(filename) % 1000,
        )

        conf = critic.max_confidence(result_img)
        if conf >= INJECTION_CONF_THRESHOLD:
            out_path = os.path.join(INJECTED_OUTPUT_DIR, filename)
            cv2.imwrite(out_path, result_img)
            # Save the injection mask (white = where polyp was injected)
            mask_out_path = os.path.join(masks_out_dir, os.path.splitext(filename)[0] + "_mask.png")
            cv2.imwrite(mask_out_path, placement_mask)
            return InjectionResult(
                filename, exemplar_row["source_image"], it, conf, placement,
                bbox[0], bbox[1], bbox[2], bbox[3], True
            )

    # exhausted iterations - save best-effort anyway, flagged
    cv2.imwrite(os.path.join(INJECTED_OUTPUT_DIR, "weak_" + filename), result_img)
    if placement_mask is not None:
        mask_out_path = os.path.join(masks_out_dir, os.path.splitext(filename)[0] + "_mask.png")
        cv2.imwrite(mask_out_path, placement_mask)
    return InjectionResult(
        filename,
        exemplar_row["source_image"] if exemplar_row is not None else "none",
        INJECTION_MAX_ITERS, conf, placement,
        bbox[0], bbox[1], bbox[2], bbox[3], False,
    )
