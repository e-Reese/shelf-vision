"""Post-hoc confidence diagnostic, never a replacement for declared validation metrics."""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from shelf_vision.training_metrics import detection_metrics
from shelf_vision.training_data import validate_dataset
from shelf_vision.media import digest

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
args = parser.parse_args()
run = Path(args.run)
experiment = json.loads((run / "experiment.json").read_text())
dataset = Path(experiment["dataset"])
validate_dataset(dataset)
if digest(experiment["checkpoint"]) != experiment["checkpoint_sha256"]:
    raise ValueError("Checkpoint changed")
frames = json.loads((dataset / "manifests/frames.json").read_text())["frames"]
truth = json.loads((dataset / "manifests/annotations.json").read_text())["annotations"]
model = YOLO(experiment["checkpoint"])
rows = []
for frame in frames:
    result = model.predict(
        frame["path"],
        imgsz=640,
        conf=0.001,
        iou=0.5,
        device=experiment["device"],
        rect=False,
        verbose=False,
    )[0]
    rows.append(
        {
            "frame_id": frame["frame_id"],
            "boxes": [
                {"bbox_xyxy": b[:4], "confidence": b[4]}
                for b in result.boxes.data.cpu().tolist()
            ],
        }
    )
diag = {
    "purpose": "Post-experiment confidence diagnostic; not a new promotion result",
    "checkpoint_sha256": experiment["checkpoint_sha256"],
    "thresholds": {
        str(t): detection_metrics(frames, truth, rows, confidence=t)
        for t in [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    },
    "max_scores": {
        r["frame_id"]: max((b["confidence"] for b in r["boxes"]), default=0)
        for r in rows
    },
}
frame = next(f for f in frames if f["split"] == "validation")
for device, rect in [("cpu", False), (experiment["device"], True)]:
    result = model.predict(
        frame["path"],
        imgsz=640,
        conf=0.001,
        iou=0.5,
        device=device,
        rect=rect,
        verbose=False,
    )[0]
    diag[f"{device}_rect_{rect}"] = {
        "frame_id": frame["frame_id"],
        "max_score": float(result.boxes.conf.max()) if len(result.boxes) else 0,
        "count": len(result.boxes),
    }
# A fresh filename preserves the original diagnostic evidence.
import tempfile

with tempfile.NamedTemporaryFile(
    mode="w", prefix="confidence-diagnostic-", suffix=".json", dir=run, delete=False
) as output:
    json.dump(diag, output, indent=2)
    print(output.name)
