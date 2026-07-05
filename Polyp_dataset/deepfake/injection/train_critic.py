"""
train_critic.py
================
One-time setup: converts your bbox CSV into YOLO-format labels and gives
you the command to train a small YOLOv8n detector, which then acts as
the critic in the refinement loop. Only needed if you don't already have
a polyp detector checkpoint.

Run this file directly:  python train_critic.py
Then train manually (needs a data.yaml + train/val split):
    from ultralytics import YOLO
    YOLO('yolov8n.pt').train(data='data.yaml', epochs=30, imgsz=640)
"""

import os
import cv2
import pandas as pd

from config import IMAGES_DIR, CSV_PATH, CRITIC_LABELS_DIR


def convert_csv_to_yolo_labels():
    df = pd.read_csv(CSV_PATH)
    os.makedirs(CRITIC_LABELS_DIR, exist_ok=True)

    n_written = 0
    for fname, group in df.groupby("filename"):
        img_path = os.path.join(IMAGES_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue
        h, w = img.shape[:2]

        lines = []
        for _, row in group.iterrows():
            x1, y1, x2, y2 = row.x1, row.y1, row.x2, row.y2
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = os.path.join(CRITIC_LABELS_DIR, os.path.splitext(fname)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        n_written += 1

    print(f"Wrote {n_written} YOLO label files to {CRITIC_LABELS_DIR}")
    print("Next: create a data.yaml with train/val image+label paths, then:")
    print("  from ultralytics import YOLO")
    print("  YOLO('yolov8n.pt').train(data='data.yaml', epochs=30, imgsz=640)")


if __name__ == "__main__":
    convert_csv_to_yolo_labels()
