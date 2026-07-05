# Polyp Removal Pipeline (modular)

Removes polyps from colonoscopy images via iterative, critic-checked
inpainting, for building a lesion-removal / medical deepfake detection
dataset.

## Structure

```
polyp_pipeline/
├── config.py            # all paths + hyperparameters - edit this first
├── masks.py              # loads your real binary masks, bbox fallback, dilation
├── critic.py              # YOLOv8 polyp detector used to judge inpainting quality
├── train_critic.py        # one-time: builds YOLO labels from your CSV
├── pipeline.py            # the self-refining loop (engine + critic agnostic)
├── main.py                # entry point - run this
├── engines/
│   ├── lama_engine.py      # primary engine (fast, texture-aware)
│   ├── sd_engine.py        # fallback engine (semantic fill, slower)
│   └── opencv_engine.py    # optional classical baseline / "easy" difficulty tier
└── requirements.txt
```

## Setup (Kaggle notebook)

1. Turn on GPU accelerator and **Internet: On** (Settings sidebar) - LaMa
   and the SD pipeline both need internet on first run to download weights.
2. `!pip install -r requirements.txt`
3. Edit `config.py`:
   - `IMAGES_DIR`, `MASKS_DIR`, `CSV_PATH` -> your dataset paths
   - `MASK_FILE_EXT` -> match your mask file extension
4. If you don't already have a polyp detector checkpoint, build the
   critic:
   ```
   python train_critic.py     # writes YOLO labels from your CSV
   # then, in a notebook cell, after making a data.yaml with a train/val split:
   from ultralytics import YOLO
   YOLO('yolov8n.pt').train(data='data.yaml', epochs=30, imgsz=640)
   ```
   Point `CRITIC_WEIGHTS` in `config.py` at the resulting `best.pt`.
5. `python main.py`

## Output

- `OUTPUT_DIR`: inpainted images, same filenames as input
  (images that never satisfied the critic are prefixed `unresolved_`)
- `LOG_CSV_PATH`: per-image log with `iterations_used`, `final_confidence`,
  `engine_used`, `mask_source`, `success` — use this to stratify your
  deepfake dataset by difficulty (e.g. images that succeeded in 1 LaMa
  pass vs. ones that needed SD fallback are meaningfully different
  "fake quality" tiers).

## Where LaMa comes from

`simple-lama-inpainting` (used in `engines/lama_engine.py`) auto-downloads
the pretrained `big-lama` checkpoint from a GitHub release mirror on first
use. Original model/repo: https://github.com/advimman/lama
(Samsung AI Center + Skoltech). No manual weight download needed.

## Injection task (adding lesions to normal / ulcerative-colitis images)

Separate entry point, same critic/config pattern as removal:

```
├── exemplar_bank.py        # crops real lesions from your removal-task dataset
├── injection_pipeline.py   # self-refining loop (critic must DETECT a lesion)
├── main_inject.py           # entry point
└── engines/
    └── kontext_gguf_engine.py   # FLUX.1 Kontext (GGUF) reference-guided injection
```

Setup:
1. Set `NORMAL_IMAGES_DIR`, `KONTEXT_GGUF_PATH` in `config.py`. Download a
   GGUF quant from https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF
   (or your Kaggle mirror) - Q4_K_M is the safest fit on a 16GB T4/P100.
2. `pip install git+https://github.com/huggingface/diffusers.git` (Kontext
   inpainting + GGUF loading need a recent build, not the pip release).
3. `python exemplar_bank.py` - crops real lesions + masks from your
   existing removal-task dataset (`IMAGES_DIR`/`MASKS_DIR`) into a
   reusable reference library.
4. `python main_inject.py`

How it works: for each normal image, a random real lesion crop + its real
mask shape is sampled from the exemplar bank, placed at a random plausible
location, and blended in via Kontext (conditioned on the reference crop,
not just a text prompt). The same critic detector used for removal then
judges the result - if it doesn't detect a lesion confidently enough, a
different exemplar/seed is tried, up to `INJECTION_MAX_ITERS`.

Known limitation: Kontext's inpainting mode doesn't always respect mask
boundaries precisely (it wasn't originally trained for masked editing).
Visually spot-check a small batch before running all 1000 images - if
placements look inconsistent, try a more generous mask margin or a higher
`KONTEXT_STRENGTH`.

## Extending

- New inpainting engine: add a class to `engines/` with an `.inpaint(img, mask)`
  method, register it in `main.py`'s `engines` dict.
- Different critic (e.g. a classifier instead of a detector): swap
  `critic.py`'s `max_confidence` implementation; `pipeline.py` doesn't
  care how the number is produced.
