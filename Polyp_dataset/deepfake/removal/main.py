"""
main.py
=======
Entry point. Run this in a Kaggle notebook cell (or `!python main.py`)
after:
  1. pip installing dependencies (see requirements.txt)
  2. setting paths in config.py
  3. training or obtaining critic weights (see train_critic.py)
"""

import os
import torch
import pandas as pd

from config import IMAGES_DIR, OUTPUT_DIR, LOG_CSV_PATH, BBOX_DIR, CRITIC_WEIGHTS
from engines import LamaEngine, SDInpaintEngine
from critic import Critic
from pipeline import refine_and_inpaint


def load_bbox_from_csv(csv_path):
    """
    Reads a per-image CSV and returns a list of (x_min, y_min, x_max, y_max) tuples.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # Remove any leading/trailing spaces
    
    bboxes = []
    for _, row in df.iterrows():
        x1 = int(row.get("x_min", row.get("xmin")))
        y1 = int(row.get("y_min", row.get("ymin")))
        x2 = int(row.get("x_max", row.get("xmax")))
        y2 = int(row.get("y_max", row.get("ymax")))
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading engines...")
    engines = {
        "lama": LamaEngine(),
        "sd": SDInpaintEngine(),
    }
    print("Loading critic...")
    critic = Critic(CRITIC_WEIGHTS)

    # iterate over all images in the images directory
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ])

    logs = []
    for i, filename in enumerate(image_files, 1):
        # look for a per-image bbox CSV with the same base name
        base = os.path.splitext(filename)[0]
        csv_path = os.path.join(BBOX_DIR, base + ".csv")

        bboxes = None
        if os.path.exists(csv_path):
            bboxes = load_bbox_from_csv(csv_path)

        res = refine_and_inpaint(
            filename, engines, critic, bboxes=bboxes
        )
        logs.append(res.__dict__)
        print(f"[{i}] {res.filename}: iters={res.iterations_used} "
              f"conf={res.final_confidence:.3f} engine={res.engine_used} "
              f"mask_source={res.mask_source} success={res.success}")

        if i % 50 == 0:
            torch.cuda.empty_cache()

    pd.DataFrame(logs).to_csv(LOG_CSV_PATH, index=False)
    n_success = sum(l["success"] for l in logs)
    print(f"\nDone. {n_success}/{len(logs)} images cleaned successfully.")
    print(f"Log written to {LOG_CSV_PATH}")


if __name__ == "__main__":
    main()

