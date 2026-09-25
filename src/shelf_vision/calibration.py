"""Development-only confidence selection, locked before validation diagnostics."""

import argparse
import json
import math
import time
from pathlib import Path
from .media import digest, save_json
from .training_data import validate_dataset
from .training_metrics import detection_metrics

THRESHOLDS = [0.001] + [i / 200 for i in range(1, 101)]
RULE = "Maximize development micro F1 at IoU 0.50; ties: precision, then threshold"
WARNING = "Development is training data; checkpoint and validation were previously examined. Existing validation is diagnostic, not independent testing. This is threshold selection, not probability calibration."


def select_threshold(frames, annotations, predictions, thresholds=THRESHOLDS):
    development = [f for f in frames if f["split"] == "development"]
    ids = {f["frame_id"] for f in development}
    labels = [a for a in annotations if a["frame_id"] in ids]
    if not development or not labels:
        raise ValueError("Require labeled development data")
    if not thresholds or any(
        not math.isfinite(t) or not 0 < t <= 1 for t in thresholds
    ):
        raise ValueError("Invalid threshold grid")
    sweep = [
        {
            "threshold": t,
            "metrics": detection_metrics(
                development, labels, predictions, confidence=t
            )["development"],
        }
        for t in thresholds
    ]
    return max(
        sweep,
        key=lambda r: (r["metrics"]["f1"], r["metrics"]["precision"], r["threshold"]),
    ), sweep


def run_calibration(run, out):
    import torch
    from ultralytics import YOLO

    run = Path(run).resolve()
    out = Path(out).resolve()
    experiment = json.loads((run / "experiment.json").read_text())
    if experiment["status"] != "complete":
        raise ValueError("Require a completed experiment")
    dataset = Path(experiment["dataset"])
    frozen = validate_dataset(dataset)
    if frozen["dataset_id"] != experiment["dataset_id"]:
        raise ValueError("Dataset differs from training experiment")
    checkpoint = Path(experiment["checkpoint"])
    if digest(checkpoint) != experiment["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    out.mkdir(parents=True, exist_ok=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    settings = {
        "imgsz": 640,
        "nms_iou": 0.5,
        "rect": False,
        "max_det": 300,
        "device": device,
    }
    save_json(
        out / "protocol.json",
        {
            "rule": RULE,
            "thresholds": THRESHOLDS,
            "settings": settings,
            "warning": WARNING,
            "checkpoint_sha256": experiment["checkpoint_sha256"],
            "dataset_id": frozen["dataset_id"],
        },
    )
    frames = json.loads((dataset / "manifests/frames.json").read_text())["frames"]
    truth = json.loads((dataset / "manifests/annotations.json").read_text())[
        "annotations"
    ]
    model = YOLO(str(checkpoint))

    def predict(rows, confidence):
        result = []
        for f in rows:
            p = model.predict(
                f["path"],
                imgsz=640,
                conf=confidence,
                iou=0.5,
                rect=False,
                max_det=300,
                device=device,
                verbose=False,
            )[0]
            result.append(
                {
                    "frame_id": f["frame_id"],
                    "boxes": [
                        {"bbox_xyxy": b[:4], "confidence": b[4]}
                        for b in p.boxes.data.cpu().tolist()
                    ],
                }
            )
        return result

    start = time.perf_counter()
    dev = [f for f in frames if f["split"] == "development"]
    dev_predictions = predict(dev, min(THRESHOLDS))
    save_json(out / "development-predictions.json", dev_predictions)
    selected, sweep = select_threshold(frames, truth, dev_predictions)
    save_json(out / "development-sweep.json", sweep)
    lock = {
        "rule": RULE,
        "threshold": selected["threshold"],
        "development": selected["metrics"],
        "checkpoint_sha256": experiment["checkpoint_sha256"],
        "dataset_id": frozen["dataset_id"],
        "experiment": str(run),
        "settings": settings,
        "warning": WARNING,
        "selection_seconds": time.perf_counter() - start,
        "sweep_sha256": digest(out / "development-sweep.json"),
        "development_predictions_sha256": digest(out / "development-predictions.json"),
    }
    save_json(out / "threshold-lock.json", lock)
    lock_hash = digest(out / "threshold-lock.json")
    print("LOCKED from development:", lock["threshold"], flush=True)
    # Only now run direct locked-threshold inference, including diagnostic validation.
    direct = predict(frames, lock["threshold"])
    save_json(out / "locked-predictions.json", direct)
    metrics = detection_metrics(frames, truth, direct, confidence=lock["threshold"])
    if any(
        metrics["development"][k] != lock["development"][k] for k in ["tp", "fp", "fn"]
    ):
        raise ValueError(
            "Direct inference differs from threshold sweep; do not apply this calibration"
        )
    validate_dataset(dataset)
    if (
        digest(checkpoint) != lock["checkpoint_sha256"]
        or digest(out / "threshold-lock.json") != lock_hash
    ):
        raise ValueError("Calibration inputs changed")
    baseline = json.loads((run / "comparison.json").read_text())["baseline"]
    report = {
        "status": "complete",
        "threshold": lock["threshold"],
        "lock_sha256": lock_hash,
        "dataset_id": frozen["dataset_id"],
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "warning": WARNING,
        "metrics": metrics,
        "baseline": baseline,
        "direct_development_parity": True,
        "total_seconds": time.perf_counter() - start,
    }
    save_json(out / "result.json", report)
    return report


def apply_calibration(root, directory):
    from .trained_detector import load_registration

    root = Path(root).resolve()
    directory = Path(directory).resolve()
    record = load_registration(root)
    result = json.loads((directory / "result.json").read_text())
    lock = json.loads((directory / "threshold-lock.json").read_text())
    if (
        result["status"] != "complete"
        or digest(directory / "threshold-lock.json") != result["lock_sha256"]
    ):
        raise ValueError("Incomplete or changed calibration")
    if any(
        record[k] != lock[k] or result[k] != lock[k]
        for k in ["dataset_id", "checkpoint_sha256"]
    ):
        raise ValueError("Calibration does not match registered model")
    if result["threshold"] != lock["threshold"] or not 0 < lock["threshold"] <= 1:
        raise ValueError("Invalid locked threshold")
    if (
        record["imgsz"] != lock["settings"]["imgsz"]
        or record["nms_iou"] != lock["settings"]["nms_iou"]
    ):
        raise ValueError("Inference settings differ from calibration")
    # Archive the exact prior configuration before updating the optional candidate.
    previous = (
        root
        / "models/registrations"
        / ("before-calibration-" + result["lock_sha256"] + ".json")
    )
    if previous.exists():
        raise FileExistsError("This calibration was already applied")
    save_json(previous, record)
    record["confidence"] = lock["threshold"]
    record["calibration"] = {
        "threshold": lock["threshold"],
        "lock_sha256": result["lock_sha256"],
        "directory": str(directory),
        "selection_split": "development",
        "independent_test_pending": True,
    }
    record["label"] = "Trained detector (calibrated pilot)"
    target = root / "models/phase-2-candidate.json"
    temporary = target.with_suffix(".tmp")
    save_json(temporary, record)
    temporary.replace(target)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--out")
    parser.add_argument("--apply")
    args = parser.parse_args()
    if args.apply:
        print(json.dumps(apply_calibration(Path.cwd(), args.apply), indent=2))
    elif args.run and args.out:
        print(json.dumps(run_calibration(args.run, args.out), indent=2))
    else:
        parser.error("Use --run and --out, or --apply")
