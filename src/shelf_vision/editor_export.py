"""Reviewed-region training export with auditable provenance."""

import io
import json
import zipfile
from PIL import Image
from .editor_store import now


def export_reviewed(store):
    output = io.BytesIO()
    images = []
    annotations = []
    rich = []
    events = []
    sources = []
    proposals = []
    reviewed = store.reviewed_snapshot()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for doc in reviewed:
            meta = doc["meta"]
            events.extend(doc["events"])
            sources.append(
                {
                    **{k: v for k, v in meta.items() if k != "path"},
                    "revision": doc["revision"],
                }
            )
            proposals.append(
                {
                    "frame_id": meta["frame_id"],
                    "original": doc["original_proposals"],
                    "runs": doc["runs"],
                }
            )
            rich.extend(doc["annotations"])
            with Image.open(meta["path"]) as original:
                for roi in meta["rois"]:
                    box = roi["bbox_xyxy"]
                    crop = original.convert("RGB").crop(box)
                    iid = len(images) + 1
                    name = f"images/{iid:06}.png"
                    buffer = io.BytesIO()
                    crop.save(buffer, format="PNG")
                    z.writestr(name, buffer.getvalue())
                    images.append(
                        {
                            "id": iid,
                            "file_name": name,
                            "width": crop.width,
                            "height": crop.height,
                            "source_frame_id": meta["frame_id"],
                            "review_roi_id": roi["roi_id"],
                            "split": meta["split"],
                            "capture_group": meta["capture_group"],
                            "source_bbox_xyxy": box,
                        }
                    )
                    for a in doc["annotations"]:
                        if a["review_roi_id"] != roi["roi_id"]:
                            continue
                        x1, y1, x2, y2 = a["bbox_xyxy"]
                        bbox = [x1 - box[0], y1 - box[1], x2 - x1, y2 - y1]
                        annotations.append(
                            {
                                "id": len(annotations) + 1,
                                "image_id": iid,
                                "category_id": 1,
                                "bbox": bbox,
                                "area": bbox[2] * bbox[3],
                                "iscrowd": 0,
                                "occupancy": a["occupancy"],
                                "identity_state": a["identity_state"],
                                "sku_id": a["observed_sku_id"],
                                "source_region_id": a["region_id"],
                            }
                        )

        def write(name, data):
            z.writestr(name, json.dumps(data, indent=2, allow_nan=False))

        write(
            "annotations.coco.json",
            {
                "info": {
                    "description": "Human-reviewed SKU display areas",
                    "created": now(),
                },
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "display_area"}],
            },
        )
        write("catalog.json", store.catalog)
        write("sources.json", sources)
        write("reviewed_annotations.json", rich)
        write("corrections.json", events)
        write("model_proposals.json", proposals)
    return {
        "bytes": output.getvalue(),
        "frame_count": len(reviewed),
        "image_count": len(images),
        "annotation_count": len(annotations),
    }
