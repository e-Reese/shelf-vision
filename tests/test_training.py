import json
import pytest
from PIL import Image
from shelf_vision.editor_store import EditorStore
from shelf_vision.training_data import freeze_dataset, validate_dataset


@pytest.fixture
def training_store(tmp_path):
    refs = tmp_path / "references"
    refs.mkdir()
    Image.new("RGB", (12, 12), "blue").save(refs / "sku-0001.jpg")
    store = EditorStore(
        tmp_path / "editor.sqlite", [{"sku_id": "sku-0001", "name": "Example"}]
    )
    for i, split in enumerate(["development", "validation"]):
        fid = f"f{i}"
        image = tmp_path / (fid + ".png")
        Image.new("RGB", (100, 80), "red").save(image)
        meta = {
            "frame_id": fid,
            "path": str(image),
            "name": fid,
            "width": 100,
            "height": 80,
            "source_id": fid,
            "capture_group": "same-visit",
            "split": split,
            "rois": [{"roi_id": fid + "-roi", "bbox_xyxy": [10, 10, 90, 70]}],
        }
        a = {
            "region_id": fid + "-a",
            "frame_id": fid,
            "review_roi_id": fid + "-roi",
            "bbox_xyxy": [20, 20, 50, 50],
            "class_name": "display_area",
            "occupancy": "occupied",
            "identity_state": "unknown",
            "observed_sku_id": None,
            "intended_sku_id": None,
            "truncated": False,
        }
        store.add_frame(meta, [a], [])
        store.save(fid, [a], 0, reviewed=True)
    return store, refs


def test_freeze_keeps_unknowns_split_and_roi_coordinates(training_store, tmp_path):
    store, refs = training_store
    out = tmp_path / "frozen"
    result = freeze_dataset(store, out, refs)
    assert result["counts"] == {
        "development": {"images": 1, "annotations": 1},
        "validation": {"images": 1, "annotations": 1},
    }
    assert result["shared_capture_groups"] == ["same-visit"]
    validate_dataset(out)
    labels = (out / "labels/train/roi-0001.txt").read_text().split()
    assert labels[0] == "0"
    assert list(map(float, labels[1:])) == pytest.approx(
        [0.3125, 0.41666667, 0.375, 0.5]
    )
    annotations = json.loads((out / "manifests/annotations.json").read_text())[
        "annotations"
    ]
    assert annotations[0]["bbox_xyxy"] == [10, 10, 40, 40]
    assert annotations[0]["identity_state"] == "unknown"
    with pytest.raises(FileExistsError):
        freeze_dataset(store, out, refs)
    (out / "images/train/roi-0001.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        validate_dataset(out)


def test_freeze_rejects_unconfirmed(training_store, tmp_path):
    store, refs = training_store
    d = store.get_frame("f0")
    store.save("f0", d["annotations"], d["revision"])
    with pytest.raises(ValueError, match="confirmed"):
        freeze_dataset(store, tmp_path / "bad", refs)


def test_freeze_rejects_source_leakage(training_store, tmp_path):
    store, refs = training_store
    with store.connection() as db:
        meta = store.get_frame("f1")["meta"]
        meta["source_id"] = "f0"
        db.execute("update frames set meta=? where id=?", (json.dumps(meta), "f1"))
    with pytest.raises(ValueError, match="multiple splits"):
        freeze_dataset(store, tmp_path / "bad", refs)


def test_freeze_excludes_final_test(training_store, tmp_path):
    store, refs = training_store
    meta = store.get_frame("f0")["meta"]
    meta = {**meta, "frame_id": "heldout", "split": "test", "source_id": "future"}
    store.add_frame(meta, [], [])
    d = freeze_dataset(store, tmp_path / "good", refs)
    assert len(d["sources"]) == 2


from shelf_vision.training_metrics import detection_metrics, promotion_gate


def test_detection_matches_by_id_counts_duplicates_unknowns_and_misses():
    frames = [
        {"frame_id": "a", "split": "validation"},
        {"frame_id": "b", "split": "validation"},
    ]
    truth = [
        {"frame_id": "a", "bbox_xyxy": [0, 0, 10, 10], "identity_state": "unknown"},
        {"frame_id": "b", "bbox_xyxy": [0, 0, 10, 10], "identity_state": "known"},
    ]
    box = {"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9}
    predictions = [
        {"frame_id": "b", "boxes": []},
        {"frame_id": "a", "boxes": [box, box]},
    ]
    m = detection_metrics(frames, truth, predictions)["validation"]
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["f1"] == 0.5
    assert m["unknown_recall"] == {"matched": 1, "total": 1, "rate": 1.0}
    assert m["known_recall"]["rate"] == 0
    with pytest.raises(ValueError, match="prediction"):
        detection_metrics(frames, truth, predictions[:1])
    with pytest.raises(ValueError, match="prediction"):
        detection_metrics(frames, truth, predictions + [predictions[0]])


def test_empty_detections_and_fixed_promotion_gate():
    m = detection_metrics(
        [{"frame_id": "a", "split": "validation"}],
        [{"frame_id": "a", "bbox_xyxy": [0, 0, 10, 10], "identity_state": "known"}],
        [{"frame_id": "a", "boxes": []}],
    )["validation"]
    assert m["recall"] == 0 and m["f1"] == 0 and m["precision"] == 0
    assert not promotion_gate(m, {"f1": 0.8, "precision": 0.4, "recall": 0.9})["passed"]
    assert promotion_gate(m, {"f1": 0.8, "precision": 0.7, "recall": 0.9})["passed"]


from shelf_vision.trained_detector import translate_roi_boxes, load_registration


def test_roi_translation_clips_boxes_and_keeps_region():
    roi = {"roi_id": "upper", "bbox_xyxy": [100, 200, 200, 300]}
    result = translate_roi_boxes(
        [[-5, 10, 40, 110, 0.8, 0], [120, 20, 140, 40, 0.9, 0]], roi
    )
    assert len(result) == 1
    assert result[0]["bbox_xyxy"] == [100, 210, 140, 300]
    assert result[0]["review_roi_id"] == "upper"


def test_registration_rejects_modified_checkpoint(tmp_path):
    from shelf_vision.media import digest

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"original")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/phase-2-candidate.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "dataset_id": "test",
                "imgsz": 640,
                "confidence": 0.25,
                "nms_iou": 0.5,
            }
        )
    )
    assert load_registration(tmp_path)["checkpoint"] == str(checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_registration(tmp_path)


def test_validation_rejects_untracked_inputs(training_store, tmp_path):
    store, refs = training_store
    out = tmp_path / "frozen"
    freeze_dataset(store, out, refs)
    (out / "images/train/extra.png").write_bytes(
        (out / "images/train/roi-0001.png").read_bytes()
    )
    with pytest.raises(ValueError, match="Unexpected"):
        validate_dataset(out)


def test_validation_rejects_relocated_absolute_paths(training_store, tmp_path):
    import shutil

    store, refs = training_store
    out = tmp_path / "frozen"
    freeze_dataset(store, out, refs)
    copied = tmp_path / "copied"
    shutil.copytree(out, copied)
    with pytest.raises(ValueError, match="path"):
        validate_dataset(copied)


@pytest.mark.parametrize(
    "malformation", ["empty_rois", "outside_roi", "missing_roi", "duplicate_roi"]
)
def test_freeze_rejects_malformed_review_scope(training_store, tmp_path, malformation):
    store, refs = training_store
    doc = store.get_frame("f1")
    meta = doc["meta"]
    annotations = doc["annotations"]
    if malformation == "empty_rois":
        meta["rois"] = []
    elif malformation == "outside_roi":
        meta["rois"][0]["bbox_xyxy"] = [-10, -10, 150, 150]
    elif malformation == "missing_roi":
        annotations[0]["review_roi_id"] = "missing"
    else:
        meta["rois"].append(dict(meta["rois"][0]))
    with store.connection() as db:
        db.execute(
            "UPDATE frames SET meta=?,annotations=? WHERE id=?",
            (json.dumps(meta), json.dumps(annotations), "f1"),
        )
    with pytest.raises(ValueError):
        freeze_dataset(store, tmp_path / "bad", refs)
    assert not (tmp_path / "bad").exists()


def test_experiment_rejects_unimplemented_operating_point():
    from shelf_vision.training import validate_operating_point

    validate_operating_point({"confidence": 0.25, "matching_iou": 0.5, "nms_iou": 0.5})
    for setting in [
        {"confidence": 0.1, "matching_iou": 0.5, "nms_iou": 0.5},
        {"confidence": 0.25, "matching_iou": 0.75, "nms_iou": 0.5},
    ]:
        with pytest.raises(ValueError, match="operating point"):
            validate_operating_point(setting)


def test_generated_label_caches_removed_before_training(training_store, tmp_path):
    from shelf_vision.training_data import clear_label_caches

    store, refs = training_store
    out = tmp_path / "frozen"
    freeze_dataset(store, out, refs)
    cache = out / "labels/train.cache"
    cache.write_bytes(b"stale")
    validate_dataset(out)
    clear_label_caches(out)
    assert not cache.exists()
    validate_dataset(out)
