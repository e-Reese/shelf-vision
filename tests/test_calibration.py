import pytest
from shelf_vision.calibration import select_threshold


def fixtures():
    frames = [
        {"frame_id": "train", "split": "development"},
        {"frame_id": "val", "split": "validation"},
    ]
    truth = [
        {"frame_id": "train", "bbox_xyxy": [0, 0, 10, 10], "identity_state": "known"},
        {"frame_id": "val", "bbox_xyxy": [0, 0, 10, 10], "identity_state": "unknown"},
    ]
    predictions = [
        {
            "frame_id": "train",
            "boxes": [
                {"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.08},
                {"bbox_xyxy": [20, 20, 30, 30], "confidence": 0.03},
            ],
        }
    ]
    return frames, truth, predictions


def test_selection_uses_only_development_and_prefers_highest_equivalent_threshold():
    frames, truth, predictions = fixtures()
    selected, sweep = select_threshold(
        frames, truth, predictions, [0.01, 0.04, 0.06, 0.1]
    )
    assert selected["threshold"] == 0.06 and selected["metrics"]["f1"] == 1
    truth[1]["bbox_xyxy"] = [90, 90, 100, 100]
    assert (
        select_threshold(frames, truth, predictions, [0.01, 0.04, 0.06, 0.1])[0]
        == selected
    )
    assert all(r["metrics"]["objects"] == 1 for r in sweep)


def test_missing_development_predictions_rejected():
    frames, truth, _ = fixtures()
    with pytest.raises(ValueError, match="prediction"):
        select_threshold(frames, truth, [], [0.01])


def test_empty_development_labels_rejected():
    frames, truth, predictions = fixtures()
    with pytest.raises(ValueError, match="development"):
        select_threshold(frames, truth[1:], predictions, [0.01])


def test_validation_predictions_not_accepted_by_selector():
    frames, truth, predictions = fixtures()
    predictions.append({"frame_id": "val", "boxes": []})
    with pytest.raises(ValueError, match="prediction"):
        select_threshold(frames, truth, predictions, [0.01])


def calibration_files(tmp_path):
    from shelf_vision.media import digest, save_json

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    record = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "dataset_id": "dataset",
        "confidence": 0.25,
        "imgsz": 640,
        "nms_iou": 0.5,
        "promotion": {"passed": False},
    }
    save_json(tmp_path / "models/phase-2-candidate.json", record)
    directory = tmp_path / "calibration"
    lock = {
        "threshold": 0.035,
        "dataset_id": "dataset",
        "checkpoint_sha256": record["checkpoint_sha256"],
        "settings": {"imgsz": 640, "nms_iou": 0.5},
    }
    save_json(directory / "threshold-lock.json", lock)
    result = {
        "status": "complete",
        "threshold": 0.035,
        "dataset_id": "dataset",
        "checkpoint_sha256": record["checkpoint_sha256"],
        "lock_sha256": digest(directory / "threshold-lock.json"),
    }
    save_json(directory / "result.json", result)
    return directory, record, result


def test_apply_archives_original_and_does_not_promote(tmp_path):
    import json
    from shelf_vision.calibration import apply_calibration

    directory, original, result = calibration_files(tmp_path)
    updated = apply_calibration(tmp_path, directory)
    assert updated["confidence"] == 0.035 and updated["promotion"] == {"passed": False}
    archived = (
        tmp_path
        / "models/registrations"
        / ("before-calibration-" + result["lock_sha256"] + ".json")
    )
    assert json.loads(archived.read_text()) == original
    with pytest.raises(FileExistsError):
        apply_calibration(tmp_path, directory)


def test_apply_rejects_different_checkpoint_without_mutation(tmp_path):
    from shelf_vision.calibration import apply_calibration
    from shelf_vision.media import save_json

    directory, original, result = calibration_files(tmp_path)
    path = tmp_path / "models/phase-2-candidate.json"
    before = path.read_bytes()
    result["checkpoint_sha256"] = "wrong"
    save_json(directory / "result.json", result)
    with pytest.raises(ValueError, match="match"):
        apply_calibration(tmp_path, directory)
    assert path.read_bytes() == before


def test_apply_rejects_changed_lock(tmp_path):
    from shelf_vision.calibration import apply_calibration

    directory, _, _ = calibration_files(tmp_path)
    with (directory / "threshold-lock.json").open("a") as file:
        file.write(" ")
    with pytest.raises(ValueError, match="changed"):
        apply_calibration(tmp_path, directory)
