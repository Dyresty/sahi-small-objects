# Small-Object Detection in Aerial Imagery with SAHI

Standard YOLO inference downsizes large images to ~640 px, so small objects (cars, people seen from a drone)
shrink to a few pixels and get missed. **SAHI** (Slicing Aided Hyper Inference) runs the detector on
overlapping tiles and merges the results, recovering those objects without retraining.

This project measures that trade-off: **how much recall SAHI buys, and what it costs in latency.**

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt
```

## Quick demo: standard vs sliced

```bash
.venv/Scripts/python scripts/compare.py
```

Downloads two SAHI demo images to `data/demo/`, runs both modes, and writes side-by-side visuals plus
`outputs/compare/results.md` (detections and ms per image). Use `--source <folder>` for your own images.

## Fine-tune on VisDrone

```bash
.venv/Scripts/python scripts/train.py                                        # full run (YOLO26s, 100 epochs, 640 px)
.venv/Scripts/python scripts/train.py --epochs 1 --fraction 0.05 --name smoke  # quick pipeline check
```

First run downloads VisDrone2019-DET (~2 GB) into `data/VisDrone/`. Weights and curves go to `runs/<name>/`.

## Roadmap

- [x] Phase 1: COCO-pretrained YOLO, standard vs SAHI on demo images
- [~] Phase 2: fine-tune YOLO on VisDrone (training script done, full run in progress) (aerial dataset, 10 classes, many tiny objects)
- [ ] Phase 3: evaluate on VisDrone val with COCO metrics (mAP, AP-small, recall) for
      standard vs SAHI vs Ultralytics native `tile`; sweep slice size and overlap
- [ ] Phase 4: results write-up (tables, plots, failure cases) + Gradio/Streamlit demo

## Results

_To be filled in after Phase 3._
