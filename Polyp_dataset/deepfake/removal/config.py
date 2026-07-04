"""
config.py
=========
Central place for every path and tunable knob. Nothing else in this
package should hardcode a path or threshold - import from here instead,
so you only ever edit one file when moving to a new Kaggle dataset.
"""

import torch

# ---------------------------------------------------------------------------
# Paths - point these at your Kaggle dataset mount
# ---------------------------------------------------------------------------

IMAGES_DIR   = "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG/images"
MASKS_DIR    = "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks"     # binary masks, same filename as images
BBOX_DIR     = "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG/bbox"  # per-image CSV files (class_name,x_min,y_min,x_max,y_max), used as fallback if mask is missing

OUTPUT_DIR   = "/kaggle/working/inpainted"
LOG_CSV_PATH = "/kaggle/working/inpainting_log.csv"

CRITIC_WEIGHTS = "/kaggle/input/models/naitikpal/yolo-polyps-model/pytorch/default/1/best (1).pt"
CRITIC_LABELS_DIR = "/kaggle/working/yolo_labels"

# ---------------------------------------------------------------------------
# Mask handling
# ---------------------------------------------------------------------------

MASK_DILATE_INIT = 3        # px - small dilation even on first pass, avoids boundary halo
MASK_DILATE_STEP = 12       # px - growth per failed refinement iteration
MASK_FILE_EXT = ".jpg"      # extension of your mask files, if different from image ext

# ---------------------------------------------------------------------------
# Refinement loop
# ---------------------------------------------------------------------------

MAX_ITERS = 4
CONF_THRESHOLD = 0.35       # critic confidence below this = "polyp successfully removed"
SWITCH_TO_SD_AFTER = 2      # after this many failed LaMa attempts, escalate to SD inpainting

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SD_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
