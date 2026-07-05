"""
main_inject.py
================
Entry point for the injection half of the dataset. Run this after:
  1. pip installing dependencies (requirements.txt + diffusers main branch)
  2. setting NORMAL_IMAGES_DIR, KONTEXT_GGUF_PATH etc. in config.py
  3. running exemplar_bank.py once to build the reference crop library
  4. having a trained critic (see train_critic.py) - same critic used for
     the removal task works here too, since it's just detecting "is a
     polyp/lesion present"
"""

import os
import torch
import pandas as pd

from config import (
    NORMAL_IMAGES_DIRS, INJECTED_OUTPUT_DIR, INJECTION_LOG_CSV_PATH,
    CRITIC_WEIGHTS,
)
from engines import KontextGGUFEngine
from critic import Critic
from exemplar_bank import load_exemplar_bank
from injection_pipeline import inject_and_refine


def main():
    os.makedirs(INJECTED_OUTPUT_DIR, exist_ok=True)

    print("Loading exemplar bank...")
    exemplar_manifest = load_exemplar_bank()

    print("Loading Kontext GGUF engine (this can take a minute)...")
    engine = KontextGGUFEngine()

    print("Loading critic...")
    critic = Critic(CRITIC_WEIGHTS)

    filenames_with_dirs = []
    for d in NORMAL_IMAGES_DIRS:
        if os.path.exists(d):
            fnames = [
                f for f in os.listdir(d)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            for f in fnames:
                filenames_with_dirs.append((d, f))

    logs = []
    for i, (target_dir, filename) in enumerate(filenames_with_dirs, 1):
        res = inject_and_refine(target_dir, filename, engine, critic, exemplar_manifest)
        logs.append(res.__dict__)
        print(f"[{i}/{len(filenames_with_dirs)}] {res.filename}: "
              f"exemplar={res.exemplar_used} iters={res.iterations_used} "
              f"conf={res.final_confidence:.3f} "
              f"bbox=({res.bbox_x1},{res.bbox_y1},{res.bbox_x2},{res.bbox_y2}) "
              f"success={res.success}")

        if i % 25 == 0:
            torch.cuda.empty_cache()

    pd.DataFrame(logs).to_csv(INJECTION_LOG_CSV_PATH, index=False)
    n_success = sum(l["success"] for l in logs)
    print(f"\nDone. {n_success}/{len(logs)} images had a convincing lesion injected.")
    print(f"Log written to {INJECTION_LOG_CSV_PATH}")


if __name__ == "__main__":
    main()
