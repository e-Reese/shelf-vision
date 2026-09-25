"""Explicit opt-in, checksum-verified pilot detector for editor proposals."""

import json
import time
from pathlib import Path
from PIL import Image
from .media import digest


def load_registration(root):
    root = Path(root).resolve()
    path = root / "models/phase-2-candidate.json"
    if not path.is_file():
        raise ValueError("No trained model has been registered yet")
    record = json.loads(path.read_text())
    checkpoint = Path(record["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(root) or not checkpoint.is_file():
        raise ValueError("Registered checkpoint is missing or outside the workspace")
    if digest(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("Registered checkpoint checksum mismatch")
    return record


def translate_roi_boxes(rows, roi):
    x, y, right, bottom = roi["bbox_xyxy"]
    result = []
    for row in rows:
        x1 = max(x, min(right, x + row[0]))
        y1 = max(y, min(bottom, y + row[1]))
        x2 = max(x, min(right, x + row[2]))
        y2 = max(y, min(bottom, y + row[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        result.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": row[4],
                "label": "display_area",
                "review_roi_id": roi["roi_id"],
                "matches": [],
            }
        )
    return result


def predict_trained(meta, registration):
    import torch
    from ultralytics import YOLO

    checkpoint = Path(registration["checkpoint"])
    if digest(checkpoint) != registration["checkpoint_sha256"]:
        raise ValueError("Registered checkpoint checksum mismatch")
    start = time.perf_counter()
    model = YOLO(str(checkpoint))
    loaded = time.perf_counter() - start
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    boxes = []
    timings = []
    with Image.open(meta["path"]) as original:
        for roi in meta["rois"]:
            crop = original.convert("RGB").crop(roi["bbox_xyxy"])
            t = time.perf_counter()
            prediction = model.predict(
                crop,
                imgsz=registration["imgsz"],
                conf=registration["confidence"],
                iou=registration["nms_iou"],
                device=device,
                rect=False,
                verbose=False,
            )[0]
            if device == "mps":
                torch.mps.synchronize()
            timings.append(
                {"roi_id": roi["roi_id"], "seconds": time.perf_counter() - t}
            )
            boxes.extend(translate_roi_boxes(prediction.boxes.data.cpu().tolist(), roi))
    return {
        "boxes": boxes,
        "provenance": {
            "model": "trained",
            "checkpoint_sha256": registration["checkpoint_sha256"],
            "dataset_id": registration["dataset_id"],
            "device": device,
            "model_load_seconds": loaded,
            "total_seconds": time.perf_counter() - start,
            "roi_timings": timings,
            "recognition": "none; detector-only proposals",
            "imgsz": registration["imgsz"],
            "confidence": registration["confidence"],
            "nms_iou": registration["nms_iou"],
            "calibration": registration.get("calibration"),
        },
    }


def register_experiment(root, run):
    from .media import save_json

    root = Path(root).resolve()
    run = Path(run).resolve()
    experiment = json.loads((run / "experiment.json").read_text())
    if experiment["status"] != "complete":
        raise ValueError("Only completed experiments can be registered")
    checkpoint = Path(experiment["checkpoint"])
    if (
        not checkpoint.resolve().is_relative_to(root)
        or digest(checkpoint) != experiment["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint checksum/path mismatch")
    config = experiment["config"]
    record = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": experiment["checkpoint_sha256"],
        "dataset_id": experiment["dataset_id"],
        "imgsz": config["train"]["imgsz"],
        "confidence": config["evaluation"]["confidence"],
        "nms_iou": config["evaluation"]["nms_iou"],
        "promotion": experiment["promotion"],
        "experiment": str(run),
        "label": "Trained detector (pilot)",
    }
    save_json(
        root / "models/registrations" / (record["checkpoint_sha256"] + ".json"), record
    )
    path = root / "models/phase-2-candidate.json"
    temporary = path.with_suffix(".tmp")
    save_json(temporary, record)
    temporary.replace(path)
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--register", required=True)
    args = parser.parse_args()
    print(json.dumps(register_experiment(Path.cwd(), args.register), indent=2))
