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

from config import OUTPUT_DIR, LOG_CSV_PATH, CSV_PATH, CRITIC_WEIGHTS
from engines import LamaEngine
from critic import Critic
from pipeline import refine_and_inpaint


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading engines...")
    engines = {
        "lama": LamaEngine(),
    }
    print("Loading critic...")
    critic = Critic(CRITIC_WEIGHTS)

    # bbox CSV is used only as a fallback for any image missing a real mask file
    bbox_df = pd.read_csv(CSV_PATH)
    bbox_lookup = {
        fname: (int(g.x1.iloc[0]), int(g.y1.iloc[0]), int(g.x2.iloc[0]), int(g.y2.iloc[0]))
        for fname, g in bbox_df.groupby("filename")
    }

    logs = []
    for i, filename in enumerate(bbox_lookup.keys(), 1):
        res = refine_and_inpaint(
            filename, engines, critic, bbox=bbox_lookup[filename]
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
