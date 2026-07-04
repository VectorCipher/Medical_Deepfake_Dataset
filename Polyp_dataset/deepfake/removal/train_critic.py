"""
train_critic.py
================
One-time setup: converts your per-image bbox CSVs into YOLO-format labels
and gives you the command to train a small YOLOv8n detector, which then
acts as the critic in the refinement loop. Only needed if you don't
already have a polyp detector checkpoint.

Each image has a corresponding CSV in BBOX_DIR with columns:
    class_name, x_min, y_min, x_max, y_max

Run this file directly:  python train_critic.py
Then train manually (needs a data.yaml + train/val split):
    from ultralytics import YOLO
    YOLO('yolov8n.pt').train(data='data.yaml', epochs=30, imgsz=640)
"""

import os
import cv2
import pandas as pd

from config import IMAGES_DIR, BBOX_DIR, CRITIC_LABELS_DIR


def convert_csvs_to_yolo_labels():
    os.makedirs(CRITIC_LABELS_DIR, exist_ok=True)

    csv_files = [f for f in os.listdir(BBOX_DIR) if f.lower().endswith(".csv")]
    n_written = 0

    for csv_file in sorted(csv_files):
        base = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(BBOX_DIR, csv_file)

        # find the matching image (try common extensions)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidate = os.path.join(IMAGES_DIR, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"WARNING: no image found for {csv_file}, skipping")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue
        h, w = img.shape[:2]

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip() # Remove any leading/trailing spaces in headers

        lines = []
        for _, row in df.iterrows():
            # Support both 'x_min' and 'xmin' naming conventions just in case
            x1 = row.get("x_min", row.get("xmin"))
            y1 = row.get("y_min", row.get("ymin"))
            x2 = row.get("x_max", row.get("xmax"))
            y2 = row.get("y_max", row.get("ymax"))
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = os.path.join(CRITIC_LABELS_DIR, base + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        n_written += 1

    print(f"Wrote {n_written} YOLO label files to {CRITIC_LABELS_DIR}")
    print("Next: create a data.yaml with train/val image+label paths, then:")
    print("  from ultralytics import YOLO")
    print("  YOLO('yolov8n.pt').train(data='data.yaml', epochs=30, imgsz=640)")


if __name__ == "__main__":
    convert_csvs_to_yolo_labels()

