"""Gradio demo: compare full-image, SAHI and Ultralytics `tile` inference on one image.

Usage:
    python app.py                       # uses runs/visdrone_yolo26s/weights/best.pt if present
    python app.py --weights yolo26n.pt  # any Ultralytics detection model
"""

import argparse
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "visdrone_yolo26s" / "weights" / "best.pt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODES = ["Full image", "SAHI", "Ultralytics tile"]


def build(weights: str):
    yolo = YOLO(weights)
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=weights, confidence_threshold=0.25, image_size=640, device=DEVICE
    )
    sahi_model.model.overrides["max_det"] = 1000  # dense aerial scenes exceed the default 300

    def run(image: np.ndarray, mode: str, slice_px: int, overlap: float, conf: float):
        if image is None:
            raise gr.Error("Upload an image first.")
        t0 = time.perf_counter()
        if mode == "Ultralytics tile":
            r = yolo.predict(image[..., ::-1], conf=conf, imgsz=640, tile=int(slice_px), max_det=1000,
                             device=DEVICE, verbose=False)[0]  # Ultralytics expects BGR arrays
            ms = (time.perf_counter() - t0) * 1000
            return r.plot(conf=False, line_width=1, font_size=10)[..., ::-1],f"**{len(r.boxes)}** detections in **{ms:.0f} ms**"
        sahi_model.confidence_threshold = conf
        if mode == "SAHI":
            res = get_sliced_prediction(image, sahi_model, slice_height=int(slice_px), slice_width=int(slice_px),
                                        overlap_height_ratio=overlap, overlap_width_ratio=overlap, verbose=0)
        else:
            res = get_prediction(image, sahi_model)
        ms = (time.perf_counter() - t0) * 1000
        vis = visualize_object_predictions(np.ascontiguousarray(image), res.object_prediction_list, hide_conf=True,
                                          rect_th=1, text_size=0.4)  # thin marks: dense scenes have 100+ boxes
        return vis["image"], f"**{len(res.object_prediction_list)}** detections in **{ms:.0f} ms**"

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "yolo26n.pt")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = ap.parse_args()
    run = build(args.weights)

    examples = sorted((ROOT / "data" / "demo").glob("*.*")) + sorted((ROOT / "data" / "VisDrone" / "images" / "val").glob("*.jpg"))[:4]
    with gr.Blocks(title="SAHI small-object detection") as demo:
        gr.Markdown(f"# Small-object detection: full image vs SAHI vs tiling\nModel: `{Path(args.weights).name}` on `{DEVICE}`")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", label="Input")
                mode = gr.Radio(MODES, value="SAHI", label="Inference mode")
                slice_px = gr.Slider(256, 1024, value=480, step=32, label="Slice / tile size (px)")
                overlap = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="SAHI overlap ratio")
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
                btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                out = gr.Image(label="Detections")
                stats = gr.Markdown()
        if examples:
            gr.Examples([[str(p)] for p in examples], inputs=[image])
        btn.click(run, [image, mode, slice_px, overlap, conf], [out, stats])
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
