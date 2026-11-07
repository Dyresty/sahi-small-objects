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
- [x] Phase 2: fine-tune YOLO26s on VisDrone (aerial dataset, 10 classes, many tiny objects)
- [ ] Phase 3: evaluate on VisDrone val with COCO metrics (mAP, AP-small, recall) for
      standard vs SAHI vs Ultralytics native `tile`; sweep slice size and overlap
- [ ] Phase 4: results write-up (tables, plots, failure cases) + Gradio/Streamlit demo

## Results

### Phase 2 baseline: YOLO26s fine-tuned on VisDrone (full-image inference, 640 px)

Trained 73 epochs (early stop, best at epoch 53, 1.5 h on an RTX 5070 Ti Laptop GPU). VisDrone2019-DET val, 548 images, 38,759 objects:

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all** | 0.525 | 0.406 | **0.403** | **0.232** |
| pedestrian | 0.569 | 0.436 | 0.456 | 0.201 |
| people | 0.551 | 0.348 | 0.358 | 0.133 |
| bicycle | 0.296 | 0.173 | 0.141 | 0.057 |
| car | 0.696 | 0.803 | 0.793 | 0.542 |
| van | 0.503 | 0.457 | 0.439 | 0.299 |
| truck | 0.609 | 0.331 | 0.363 | 0.234 |
| tricycle | 0.447 | 0.301 | 0.309 | 0.162 |
| awning-tricycle | 0.376 | 0.179 | 0.166 | 0.102 |
| bus | 0.709 | 0.522 | 0.537 | 0.385 |
| motor | 0.496 | 0.514 | 0.472 | 0.204 |

Low recall on tiny classes (people, bicycle, awning-tricycle) is what Phase 3 tests SAHI against.
