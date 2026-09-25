"""Freeze confirmed editor labels, ROI crops and references for reproducible training."""

import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from .editor_export import export_reviewed
from .media import digest, save_json
from .schema import validate_annotation
from .catalog import validate_box


def validate_review_scope(records, catalog):
    ids = {p["sku_id"] for p in catalog}
    for d in records:
        m = d["meta"]
        rois = {}
        seen = set()
        with Image.open(m["path"]) as im:
            if im.size != (m["width"], m["height"]):
                raise ValueError("Source image dimensions disagree")
        if not m["rois"]:
            raise ValueError("Source has no review ROIs")
        for roi in m["rois"]:
            rid = roi["roi_id"]
            box = roi["bbox_xyxy"]
            if not rid or rid in rois:
                raise ValueError("Duplicate or missing ROI ID")
            validate_box(box, m["width"], m["height"])
            if any(float(v) != int(v) for v in box):
                raise ValueError("Review ROI bounds must be integer pixels")
            rois[rid] = box
        for a in d["annotations"]:
            if a["frame_id"] != m["frame_id"] or a["review_roi_id"] not in rois:
                raise ValueError("Annotation has invalid frame or ROI association")
            if not a["region_id"] or a["region_id"] in seen:
                raise ValueError("Duplicate or missing annotation ID")
            seen.add(a["region_id"])
            validate_annotation(a, ids, m["width"], m["height"])
            b = a["bbox_xyxy"]
            r = rois[a["review_roi_id"]]
            if not (r[0] <= b[0] < b[2] <= r[2] and r[1] <= b[1] < b[3] <= r[3]):
                raise ValueError("Annotation outside its ROI")


def clear_label_caches(path):
    # Ultralytics caches parsed labels even when its image cache is disabled.
    # Force regeneration from the hashed text labels before each training run.
    for name in ["train.cache", "val.cache"]:
        (Path(path) / "labels" / name).unlink(missing_ok=True)


def freeze_dataset(store, out, catalog_dir):
    out = Path(out).resolve()
    catalog_dir = Path(catalog_dir)
    if out.exists():
        raise FileExistsError(out)
    records = store.reviewed_snapshot()
    current = [f for f in store.list_frames() if f["split"] != "test"]
    frozen = {d["meta"]["frame_id"]: d["revision"] for d in records}
    if not current or any(
        f["status"] != "reviewed" or frozen.get(f["frame_id"]) != f["revision"]
        for f in current
    ):
        raise ValueError(
            "All non-test frames must be confirmed and stable before freezing"
        )
    validate_review_scope(records, store.catalog)
    source_splits = {}
    groups = defaultdict(set)
    for d in records:
        m = d["meta"]
        split = m["split"]
        source = m.get("video_id") or m["source_id"]
        if split not in {"development", "validation"}:
            raise ValueError("Invalid training split")
        if source_splits.setdefault(source, split) != split:
            raise ValueError("A source appears in multiple splits")
        groups[m["capture_group"]].add(split)
    if set(source_splits.values()) != {"development", "validation"}:
        raise ValueError("Require development and validation sources")
    result = export_reviewed(
        SimpleNamespace(catalog=store.catalog, reviewed_snapshot=lambda: records)
    )
    if result["annotation_count"] != sum(len(d["annotations"]) for d in records):
        raise ValueError("Export dropped annotations")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".freeze-", dir=out.parent))
    try:
        (tmp / "reviewed.zip").write_bytes(result["bytes"])
        frames = []
        areas = []
        rois = []
        counts = defaultdict(Counter)
        with zipfile.ZipFile(io.BytesIO(result["bytes"])) as z:
            coco = json.loads(z.read("annotations.coco.json"))
            sources = {s["frame_id"]: s for s in json.loads(z.read("sources.json"))}
            rich = {
                (a["frame_id"], a["region_id"]): a
                for a in json.loads(z.read("reviewed_annotations.json"))
            }
            ids = {p["sku_id"] for p in store.catalog}
            for i, im in enumerate(coco["images"], 1):
                split = im["split"]
                role = "train" if split == "development" else "val"
                fid = f"roi-{i:04d}"
                for kind in ["images", "labels"]:
                    (tmp / kind / role).mkdir(parents=True, exist_ok=True)
                relative = Path("images") / role / (fid + ".png")
                (tmp / relative).write_bytes(z.read(im["file_name"]))
                with Image.open(tmp / relative) as image:
                    if image.size != (im["width"], im["height"]):
                        raise ValueError("Crop size mismatch")
                source = sources[im["source_frame_id"]]
                frame = {
                    **im,
                    "frame_id": fid,
                    "path": str(out / relative),
                    "video_id": source.get("video_id") or source["source_id"],
                    "source_id": source["source_id"],
                }
                frames.append(frame)
                roi = {
                    "roi_id": fid + "-roi",
                    "frame_id": fid,
                    "bbox_xyxy": [0, 0, im["width"], im["height"]],
                }
                rois.append(roi)
                lines = []
                for a in coco["annotations"]:
                    if a["image_id"] != im["id"]:
                        continue
                    x, y, w, h = a["bbox"]
                    original = rich[(im["source_frame_id"], a["source_region_id"])]
                    label = {
                        **original,
                        "source_region_id": original["region_id"],
                        "region_id": fid + "-" + str(a["id"]),
                        "frame_id": fid,
                        "review_roi_id": roi["roi_id"],
                        "bbox_xyxy": [x, y, x + w, y + h],
                    }
                    validate_annotation(label, ids, im["width"], im["height"])
                    areas.append(label)
                    lines.append(
                        "0 "
                        + " ".join(
                            f"{v:.8f}"
                            for v in [
                                (x + w / 2) / im["width"],
                                (y + h / 2) / im["height"],
                                w / im["width"],
                                h / im["height"],
                            ]
                        )
                    )
                (tmp / "labels" / role / (fid + ".txt")).write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
                counts[split]["images"] += 1
                counts[split]["annotations"] += len(lines)
        if any(counts[split]["images"] == 0 for split in ["development", "validation"]):
            raise ValueError("Require actual crops in both splits")
        if counts["development"]["annotations"] == 0:
            raise ValueError("No training annotations")
        (tmp / "references").mkdir()
        for p in store.catalog:
            shutil.copyfile(
                catalog_dir / (p["sku_id"] + ".jpg"),
                tmp / "references" / (p["sku_id"] + ".jpg"),
            )
        save_json(
            tmp / "manifests/frames.json", {"schema_version": 1, "frames": frames}
        )
        save_json(
            tmp / "manifests/annotations.json",
            {
                "schema_version": 1,
                "review_status": "human_reviewed",
                "human_review_status": "reviewed",
                "annotations": areas,
                "rois": rois,
            },
        )
        save_json(
            tmp / "manifests/catalog.json",
            {"schema_version": 1, "products": store.catalog},
        )
        (tmp / "data.yaml").write_text(
            f"path: {json.dumps(str(out))}\ntrain: images/train\nval: images/val\nnames:\n  0: display_area\n"
        )
        files = {
            str(p.relative_to(tmp)): digest(p)
            for p in sorted(tmp.rglob("*"))
            if p.is_file()
        }
        record = {
            "schema_version": 1,
            "counts": dict(counts),
            "sources": list(sources.values()),
            "shared_capture_groups": sorted(k for k, v in groups.items() if len(v) > 1),
            "files": files,
        }
        record["dataset_id"] = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()
        save_json(tmp / "dataset.json", record)
        tmp.rename(out)
        return record
    except Exception:
        shutil.rmtree(tmp)
        raise


def validate_dataset(path):
    path = Path(path).resolve()
    record = json.loads((path / "dataset.json").read_text())
    expected = record["dataset_id"]
    payload = {k: v for k, v in record.items() if k != "dataset_id"}
    if (
        hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        != expected
    ):
        raise ValueError("Dataset manifest hash mismatch")
    for relative, expected_hash in record["files"].items():
        file = (path / relative).resolve()
        if (
            not file.is_relative_to(path)
            or not file.is_file()
            or digest(file) != expected_hash
        ):
            raise ValueError("Dataset file hash mismatch: " + relative)
    allowed = set(record["files"]) | {
        "dataset.json",
        "labels/train.cache",
        "labels/val.cache",
    }
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if actual - allowed:
        raise ValueError(
            "Unexpected dataset files: " + ", ".join(sorted(actual - allowed))
        )
    import yaml

    config = yaml.safe_load((path / "data.yaml").read_text())
    if (
        Path(config["path"]).resolve() != path
        or config["train"] != "images/train"
        or config["val"] != "images/val"
    ):
        raise ValueError("Dataset YAML path is outside validated snapshot")
    frames = json.loads((path / "manifests/frames.json").read_text())["frames"]
    for frame in frames:
        image = Path(frame["path"]).resolve()
        role = "train" if frame["split"] == "development" else "val"
        expected = path / "images" / role / (frame["frame_id"] + ".png")
        if image != expected:
            raise ValueError("Dataset frame path is outside validated snapshot")
    if {str(Path(f["path"]).relative_to(path)) for f in frames} != {
        p for p in record["files"] if p.startswith("images/")
    }:
        raise ValueError("Dataset frame paths disagree with image inventory")
    return record


if __name__ == "__main__":
    import argparse
    from .editor_store import EditorStore

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    catalog = json.loads(Path("data/manifests/catalog.json").read_text())["products"]
    store = EditorStore("data/editor/editor.sqlite", catalog)
    print(json.dumps(freeze_dataset(store, args.out, "data/derived/catalog"), indent=2))
