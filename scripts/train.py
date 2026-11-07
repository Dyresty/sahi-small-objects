"""Fine-tune YOLO on VisDrone2019-DET (aerial, 10 classes, mostly tiny objects).

The dataset (~2 GB) downloads to data/VisDrone on first run, kept inside this project
instead of the global Ultralytics datasets_dir.

Usage:
    python scripts/train.py                              # full run
    python scripts/train.py --epochs 1 --fraction 0.05 --name smoke   # pipeline check
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

ROOT = Path(__file__).resolve().parents[1]


def data_yaml() -> str:
    cfg = YAML.load(check_yaml("VisDrone.yaml"))
    cfg["path"] = str(ROOT / "data" / "VisDrone")  # absolute, so Ultralytics won't use its global datasets_dir
    out = ROOT / "data" / "VisDrone.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(out, cfg)
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
    ap.add_argument("--name", default="visdrone_yolo26s")
    args = ap.parse_args()

    YOLO(args.model).train(
        data=data_yaml(),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        patience=20,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        workers=4,  # Windows dataloader workers are spawned processes; keep modest
    )


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
