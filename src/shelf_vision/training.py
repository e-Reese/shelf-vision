"""Single declared local experiment: frozen baseline, training, fixed comparison."""

import argparse
import html
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from PIL import Image, ImageDraw
from .baseline import run_baseline
from .evaluate import recognition_metrics
from .media import digest, save_json
from .training_data import validate_dataset, clear_label_caches
from .training_metrics import detection_metrics, promotion_gate


def visual_report(out, frames, truth, baseline, candidate, comparison):
    panels = []
    for frame in frames:
        if frame["split"] != "validation":
            continue
        fid = frame["frame_id"]
        images = []
        for name, rows in [("baseline", baseline), ("trained", candidate)]:
            im = Image.open(frame["path"]).convert("RGB")
            draw = ImageDraw.Draw(im)
            for a in [a for a in truth if a["frame_id"] == fid]:
                draw.rectangle(a["bbox_xyxy"], outline="#57f59a", width=5)
            for p in next(r["boxes"] for r in rows if r["frame_id"] == fid):
                if p["confidence"] >= 0.25:
                    draw.rectangle(p["bbox_xyxy"], outline="#ff7755", width=3)
            im.thumbnail((900, 900))
            filename = f"{fid}-{name}.jpg"
            im.save(out / filename, quality=92)
            images.append(
                f'<figure><figcaption>{name}</figcaption><img src="{filename}"></figure>'
            )
        panels.append(
            "<section><h2>"
            + html.escape(fid + " · " + frame["source_frame_id"])
            + "</h2><div>"
            + "".join(images)
            + "</div></section>"
        )
    text = json.dumps(
        {
            k: v
            for k, v in comparison.items()
            if k in ["baseline", "trained", "promotion"]
        },
        indent=2,
    )
    (out / "validation.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Reviewed detector comparison</title><style>body{font:15px system-ui;background:#15271e;color:#e2ebdf;margin:30px}section>div{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%}figcaption{padding:8px}pre{white-space:pre-wrap}h2{font-size:17px}</style><h1>Validation comparison</h1><p>Same-visit pilot. Green: human truth. Orange: predictions at confidence ≥ 0.25. Both models see these same ROI crops.</p>'
        + "".join(panels)
        + "<details><summary>Metrics and denominators</summary><pre>"
        + html.escape(text)
        + "</pre></details>"
    )


def validate_operating_point(settings):
    if settings.get("confidence") != 0.25 or settings.get("matching_iou") != 0.5:
        raise ValueError(
            "This experiment supports only the declared operating point: confidence 0.25, matching IoU 0.50"
        )


def run_experiment(dataset, out, config_path):
    import torch
    from ultralytics import YOLO

    dataset = Path(dataset).resolve()
    out = Path(out).resolve()
    config_path = Path(config_path).resolve()
    frozen = validate_dataset(dataset)
    config = json.loads(config_path.read_text())
    validate_operating_point(config["evaluation"])
    root = Path(__file__).resolve().parents[2]
    initial = (root / config["initial_checkpoint"]).resolve()
    if not initial.exists():
        raise ValueError("Initial checkpoint must already exist locally")
    out.mkdir(parents=True, exist_ok=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    record = {
        "dataset_id": frozen["dataset_id"],
        "dataset": str(dataset),
        "config": config,
        "initial_checkpoint_sha256": digest(initial),
        "device": device,
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["torch", "ultralytics", "transformers", "numpy"]
        },
        "status": "running",
        "warning": "Same-visit grouped validation; not independent new-visit performance.",
    }
    save_json(out / "experiment.json", record)
    try:
        os.environ.setdefault("HF_HOME", str(root / "models/huggingface"))
        print("PHASE: frozen ROI baseline", flush=True)
        run_baseline(
            root / "configs/phase-0-baseline.json",
            dataset / "manifests",
            out / "baseline",
            gallery_dir=dataset / "references",
        )
        # Verify the frozen inputs again before optimizing any weights.
        validate_dataset(dataset)
        clear_label_caches(dataset)
        print("PHASE: supervised training", flush=True)
        model = YOLO(str(initial))
        start = time.perf_counter()
        model.train(
            data=str(dataset / "data.yaml"),
            device=device,
            project=str(out),
            name="train",
            exist_ok=False,
            plots=False,
            **config["train"],
        )
        record["training_seconds"] = time.perf_counter() - start
        checkpoint = out / "train/weights/best.pt"
        record["checkpoint"] = str(checkpoint)
        record["checkpoint_sha256"] = digest(checkpoint)
        save_json(out / "experiment.json", record)
        print("PHASE: fixed operating-point comparison", flush=True)
        trained = YOLO(str(checkpoint))
        frames = json.loads((dataset / "manifests/frames.json").read_text())["frames"]
        truth = json.loads((dataset / "manifests/annotations.json").read_text())[
            "annotations"
        ]
        # Warm-up is excluded from per-image wall timings, retained separately.
        start = time.perf_counter()
        trained.predict(
            frames[0]["path"],
            imgsz=config["train"]["imgsz"],
            device=device,
            conf=0.25,
            rect=False,
            verbose=False,
        )
        record["prediction_warmup_seconds"] = time.perf_counter() - start
        predictions = []
        for f in frames:
            if device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            result = trained.predict(
                f["path"],
                imgsz=config["train"]["imgsz"],
                device=device,
                conf=config["evaluation"]["confidence"],
                iou=config["evaluation"]["nms_iou"],
                rect=False,
                verbose=False,
            )[0]
            if device == "mps":
                torch.mps.synchronize()
            predictions.append(
                {
                    "frame_id": f["frame_id"],
                    "inference_seconds": time.perf_counter() - start,
                    "boxes": [
                        {"bbox_xyxy": r[:4], "confidence": r[4]}
                        for r in result.boxes.data.cpu().tolist()
                    ],
                }
            )
        save_json(
            out / "trained-predictions.json",
            {"frames": predictions, "checkpoint_sha256": record["checkpoint_sha256"]},
        )
        baseline = json.loads((out / "baseline/predictions.json").read_text())
        bm = detection_metrics(frames, truth, baseline["frames"])
        tm = detection_metrics(frames, truth, predictions)
        crops = [c for c in baseline["crop_results"] if c["crop_kind"] == "area"]
        byid = {f["frame_id"]: f for f in frames}
        dev = [a for a in truth if byid[a["frame_id"]]["split"] == "development"]
        sweep = [
            {"threshold": t / 100, **recognition_metrics(dev, crops, t / 100)}
            for t in range(0, 101, 5)
        ]
        eligible = [
            r
            for r in sweep
            if r["unknown_false_accept"]["rate"] is not None
            and r["unknown_false_accept"]["rate"] <= 0.1
        ]
        threshold = (
            max(
                eligible,
                key=lambda r: (r["known_acceptance"]["rate"] or 0, -r["threshold"]),
            )["threshold"]
            if eligible
            else None
        )
        retrieval = {
            split: recognition_metrics(
                [a for a in truth if byid[a["frame_id"]]["split"] == split],
                crops,
                threshold if threshold is not None else 1.1,
            )
            for split in ["development", "validation"]
        }
        comparison = {
            "dataset_id": frozen["dataset_id"],
            "warning": record["warning"],
            "operating_point": config["evaluation"],
            "baseline": bm,
            "trained": tm,
            "promotion": promotion_gate(bm["validation"], tm["validation"]),
            "recognition_baseline": retrieval,
            "rejection_threshold": threshold,
            "threshold_selection": "development only; unknown false acceptance <= 0.10; maximize known acceptance",
            "development_threshold_sweep": sweep,
        }
        save_json(out / "comparison.json", comparison)
        visual_report(out, frames, truth, baseline["frames"], predictions, comparison)
        validate_dataset(dataset)
        record["status"] = "complete"
        record["promotion"] = comparison["promotion"]
        save_json(out / "experiment.json", record)
        return comparison
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        save_json(out / "experiment.json", record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/phase-2-training.json")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.dataset, args.out, args.config), indent=2))
