"""Compare standard (full-image) vs SAHI sliced inference on a folder of images.

Writes side-by-side visuals and a results table to outputs/compare/.

Usage:
    python scripts/compare.py --source data/demo --model yolo26n.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.file import download_from_url

DEMO_URLS = [
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def fetch_demo(dst: Path) -> None:
    for url in DEMO_URLS:
        path = dst / url.rsplit("/", 1)[-1]
        if not path.exists():
            download_from_url(url, str(path))


def visual(result) -> Image.Image:
    """Render a PredictionResult to a PIL image."""
    from sahi.utils.cv import visualize_object_predictions

    out = visualize_object_predictions(
        np.ascontiguousarray(result.image), result.object_prediction_list, hide_conf=True
    )
    return Image.fromarray(out["image"])


def side_by_side(a: Image.Image, b: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/demo")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--slice", type=int, default=256)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/compare")
    args = ap.parse_args()

    src, out = Path(args.source), Path(args.out)
    src.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if args.source == "data/demo":
        fetch_demo(src)

    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images in {src}")

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        confidence_threshold=args.conf,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    get_prediction(str(images[0]), model)  # warm-up so first timing is fair

    rows = []
    for img in images:
        t0 = time.perf_counter()
        std = get_prediction(str(img), model)
        t1 = time.perf_counter()
        sliced = get_sliced_prediction(
            str(img),
            model,
            slice_height=args.slice,
            slice_width=args.slice,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            verbose=0,
        )
        t2 = time.perf_counter()

        side_by_side(visual(std), visual(sliced)).save(out / f"{img.stem}_std_vs_sahi.jpg", quality=90)
        rows.append((img.name, len(std.object_prediction_list), len(sliced.object_prediction_list),
                     (t1 - t0) * 1000, (t2 - t1) * 1000))

    table = ["| image | standard dets | SAHI dets | standard ms | SAHI ms |", "|---|---|---|---|---|"]
    table += [f"| {n} | {a} | {b} | {ta:.0f} | {tb:.0f} |" for n, a, b, ta, tb in rows]
    (out / "results.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"\nVisuals: {out.resolve()}")


if __name__ == "__main__":
    main()
