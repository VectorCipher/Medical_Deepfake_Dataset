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
import shutil
import torch
import pandas as pd

from config import IMAGES_DIR, OUTPUT_DIR, OUTPUT_BBOX_DIR, LOG_CSV_PATH, BBOX_DIR, CRITIC_WEIGHTS
from engines import LamaEngine, OpenCVEngine
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
    os.makedirs(OUTPUT_BBOX_DIR, exist_ok=True)

    print("Loading engines...")
    engines = {
        "lama": LamaEngine(),
        "opencv": OpenCVEngine(),
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
        
        # Save the bounding box CSV with the matching filename (prefixing if unresolved)
        if os.path.exists(csv_path):
            out_base = "unresolved_" + base if not res.success else base
            shutil.copy(csv_path, os.path.join(OUTPUT_BBOX_DIR, out_base + ".csv"))

        print(f"[{i}] {res.filename}: iters={res.iterations_used} "
              f"conf={res.final_confidence:.3f} engine={res.engine_used} "
              f"mask_source={res.mask_source} success={res.success}")

        if i % 50 == 0:
            torch.cuda.empty_cache()

    pd.DataFrame(logs).to_csv(LOG_CSV_PATH, index=False)
    n_success = sum(l["success"] for l in logs)
    print(f"\nDone. {n_success}/{len(logs)} images cleaned successfully.")
    print(f"Log written to {LOG_CSV_PATH}")
    print(f"Inpainted images saved to {OUTPUT_DIR}")
    print(f"Corresponding bboxes saved to {OUTPUT_BBOX_DIR}")


if __name__ == "__main__":
    main()

