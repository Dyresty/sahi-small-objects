"""Evaluate full-image vs SAHI vs Ultralytics `tile` inference on VisDrone with COCO metrics.

Every mode is scored with the same pycocotools evaluation (maxDets=1000, VisDrone images
hold up to ~900 objects), so numbers are directly comparable.

Modes:  standard (full image via SAHI wrapper) | yolo (plain Ultralytics predict, cross-check)
        | sahi:<slice px> | tile:<tile px>
Usage:
    python scripts/evaluate.py --limit 20                     # quick check
    python scripts/evaluate.py --modes standard,sahi:640,tile:640
"""

import argparse
import contextlib
import csv
import io
import json
import time
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "VisDrone"
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
MAX_DETS = 1000
METRICS = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1000", "AR_small", "AR_medium", "AR_large"]


def build_gt(images: list[Path], out: Path) -> Path:
    """Convert YOLO txt labels to a COCO GT json (category id = class + 1)."""
    imgs, anns = [], []
    for i, p in enumerate(images):
        w, h = Image.open(p).size
        imgs.append({"id": i, "file_name": p.name, "width": w, "height": h})
        lbl = DATA / "labels" / p.parent.name / f"{p.stem}.txt"
        for line in lbl.read_text().splitlines() if lbl.exists() else []:
            c, xc, yc, bw, bh = map(float, line.split())
            bw, bh = bw * w, bh * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(c) + 1, "iscrowd": 0,
                         "bbox": [xc * w - bw / 2, yc * h - bh / 2, bw, bh], "area": bw * bh})
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)]
    out.write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}))
    return out


def run_sahi(weights: str, images: list[Path], conf: float, slice_px: int | None, overlap: float):
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=weights, confidence_threshold=conf, image_size=640,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    model.model.overrides["max_det"] = MAX_DETS  # SAHI doesn't pass max_det; default 300 would cap recall

    def predict(p):
        if slice_px is None:
            return get_prediction(str(p), model)
        return get_sliced_prediction(str(p), model, slice_height=slice_px, slice_width=slice_px,
                                     overlap_height_ratio=overlap, overlap_width_ratio=overlap, verbose=0)

    predict(images[0])  # warm-up
    dets, t = [], 0.0
    for i, p in enumerate(images):
        t0 = time.perf_counter()
        res = predict(p)
        t += time.perf_counter() - t0
        for o in res.object_prediction_list:
            x1, y1, x2, y2 = map(float, o.bbox.to_xyxy())  # SAHI gives numpy floats; json needs Python floats
            dets.append({"image_id": i, "category_id": o.category.id + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(o.score.value)})
    return dets, t / len(images) * 1000


def run_tile(weights: str, images: list[Path], conf: float, tile_px: int | bool):
    model = YOLO(weights)
    kw = dict(conf=conf, imgsz=640, tile=tile_px, max_det=MAX_DETS, verbose=False)
    model.predict(str(images[0]), **kw)  # warm-up
    dets, t = [], 0.0
    for i, p in enumerate(images):
        t0 = time.perf_counter()
        r = model.predict(str(p), **kw)[0]
        t += time.perf_counter() - t0
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            dets.append({"image_id": i, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    return dets, t / len(images) * 1000


def coco_eval(gt: COCO, dets: list[dict]) -> list[float]:
    if not dets:
        return [0.0] * len(METRICS)
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.maxDets = [10, 100, MAX_DETS]  # summarize() reports AP at maxDets[2]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # stats: 0-5 AP@[maxDets[2]], 6-8 AR@maxDets, 9-11 AR by size; keep AR@1000 and AR by size
    s = ev.stats
    return [s[0], s[1], s[2], s[3], s[4], s[5], s[8], s[9], s[10], s[11]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "visdrone_yolo26s" / "weights" / "best.pt"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--modes", default="standard,yolo,sahi:320,sahi:480,sahi:640,sahi:800,sahi:960,tile:480,tile:640,tile:800,tile:960")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = sorted((DATA / "images" / args.split).glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(build_gt(images, out / f"gt_{args.split}.json")))

    rows = []
    for mode in args.modes.split(","):
        kind, _, size = mode.partition(":")
        if kind == "standard":
            dets, ms = run_sahi(args.weights, images, args.conf, None, args.overlap)
        elif kind == "sahi":
            dets, ms = run_sahi(args.weights, images, args.conf, int(size), args.overlap)
        elif kind == "yolo":
            dets, ms = run_tile(args.weights, images, args.conf, False)
        elif kind == "tile":
            dets, ms = run_tile(args.weights, images, args.conf, int(size))
        else:
            raise SystemExit(f"Unknown mode: {mode}")
        (out / f"dets_{mode.replace(':', '_')}.json").write_text(json.dumps(dets))
        rows.append([mode, len(images), round(ms, 1)] + [round(v, 4) for v in coco_eval(gt, dets)])
        print(f"{mode:>10}  ms/img={ms:7.1f}  " + "  ".join(f"{k}={v:.3f}" for k, v in zip(METRICS, rows[-1][3:])))

    name = f"results_{args.split}" + (f"_n{args.limit}" if args.limit else "") + ".csv"
    with open(out / name, "w", newline="") as f:
        csv.writer(f).writerows([["mode", "images", "ms_per_img"] + METRICS] + rows)
    print(f"\nSaved {out / name}")


if __name__ == "__main__":
    main()
