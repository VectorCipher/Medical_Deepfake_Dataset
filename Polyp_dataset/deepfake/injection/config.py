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
BBOX_DIR     = "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG/bbox"  # directory with csv files for every image (used as fallback if mask missing)

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


# ---------------------------------------------------------------------------
# Injection task (adding a lesion to normal images)
# ---------------------------------------------------------------------------

NORMAL_IMAGES_DIRS = [
    "/kaggle/input/datasets/francismon/curated-colon-dataset-for-deep-learning/train/0_normal",
]
EXEMPLAR_BANK_DIR = "/kaggle/working/exemplar_bank"                     # cropped real lesions, built once
INJECTED_OUTPUT_DIR = "/kaggle/working/injected"
INJECTION_LOG_CSV_PATH = "/kaggle/working/injection_log.csv"

# GGUF-quantized Kontext transformer file - download the .gguf from
# QuantStack/FLUX.1-Kontext-dev-GGUF (or the Kaggle mirror you're using)
# and set the path here. Pick a quant level that fits your GPU:
#   Q4_K_M ~7GB  - safest on a 16GB T4/P100 alongside other loaded models
#   Q5_K_M ~8.5GB
#   Q6_K   ~10GB - better quality, less headroom for the critic model
KONTEXT_GGUF_PATH = "/kaggle/input/models/tungnguyen62cg/flux.1-kontext-dev-gguf/gguf/default/1/flux1-kontext-dev-Q8_0.gguf"

# base repo supplies VAE + text encoders (CLIP-L, T5-XXL) + config,
# NOT included in the GGUF file, which only contains the transformer
KONTEXT_BASE_REPO = "black-forest-labs/FLUX.1-Kontext-dev"

KONTEXT_NUM_STEPS = 28
KONTEXT_GUIDANCE_SCALE = 3.5
KONTEXT_STRENGTH = 1.0          # 1.0 = fully regenerate masked region

INJECTION_MAX_ITERS = 3
INJECTION_CONF_THRESHOLD = 0.5   # critic confidence ABOVE this = "lesion convincingly present"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

