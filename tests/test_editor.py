import io
import json
import zipfile
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from shelf_vision.editor_store import EditorStore, Conflict
from shelf_vision.editor_api import create_app


def area(**updates):
    result = {
        "region_id": "a1",
        "frame_id": "f1",
        "review_roi_id": "roi1",
        "bbox_xyxy": [10, 10, 30, 30],
        "class_name": "display_area",
        "occupancy": "occupied",
        "identity_state": "known",
        "observed_sku_id": "sku-0001",
        "intended_sku_id": None,
        "truncated": False,
    }
    result.update(updates)
    return result


@pytest.fixture
def store(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    path = images / "source.png"
    Image.new("RGB", (100, 80), "red").save(path)
    s = EditorStore(
        tmp_path / "editor.sqlite",
        catalog=[{"sku_id": "sku-0001", "name": "Example product"}],
    )
    s.add_frame(
        {
            "frame_id": "f1",
            "path": str(path),
            "width": 100,
            "height": 80,
            "source_id": "source",
            "name": "Example.png",
            "capture_group": "visit1",
            "split": "development",
            "rois": [{"roi_id": "roi1", "bbox_xyxy": [10, 10, 80, 70]}],
        },
        [area()],
        [
            {
                "bbox_xyxy": [11, 11, 31, 31],
                "confidence": 0.8,
                "matches": [{"sku_id": "sku-0001", "score": 0.7}],
            }
        ],
    )
    return s


def test_seed_is_idempotent_and_saves_preserve_proposals(store):
    doc = store.get_frame("f1")
    original = doc["proposals"]
    updated = area(bbox_xyxy=[12, 12, 34, 34])
    saved = store.save("f1", [updated], 0, reviewed=True)
    assert saved["revision"] == 1 and saved["status"] == "reviewed"
    store.add_frame(doc["meta"], [area()], [])
    restarted = EditorStore(store.path, catalog=store.catalog)
    assert restarted.get_frame("f1")["annotations"][0]["bbox_xyxy"] == [12, 12, 34, 34]
    assert restarted.get_frame("f1")["proposals"] == original
    assert len(restarted.events("f1")) == 1
    reopened = store.save("f1", [area()], 1)
    assert reopened["status"] == "in_progress"


def test_stale_save_fails_without_overwriting(store):
    store.save("f1", [area()], 0)
    with pytest.raises(Conflict):
        store.save("f1", [], 0)
    assert len(store.get_frame("f1")["annotations"]) == 1
    assert len(store.events("f1")) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"occupancy": "empty"},
        {"bbox_xyxy": [0, 0, 40, 40]},
        {"region_id": ""},
        {"observed_sku_id": "missing"},
    ],
)
def test_invalid_annotations_never_persist(store, changes):
    with pytest.raises(ValueError):
        store.save("f1", [area(**changes)], 0)
    assert store.get_frame("f1")["revision"] == 0


def test_reviewed_export_is_roi_relative_and_excludes_test(store, tmp_path):
    from shelf_vision.editor_export import export_reviewed

    assert export_reviewed(store)["frame_count"] == 0
    store.save("f1", [area()], 0, reviewed=True)
    test = store.get_frame("f1")["meta"]
    test = {**test, "frame_id": "test1", "split": "test", "capture_group": "final"}
    test["rois"] = [{"roi_id": "test-roi", "bbox_xyxy": [0, 0, 100, 80]}]
    store.add_frame(test, [], [])
    store.save("test1", [], 0, reviewed=True)
    result = export_reviewed(store)
    with zipfile.ZipFile(io.BytesIO(result["bytes"])) as z:
        coco = json.loads(z.read("annotations.coco.json"))
        assert len(coco["images"]) == 1
        assert coco["images"][0]["width"] == 70 and coco["images"][0]["height"] == 60
        assert coco["annotations"][0]["bbox"] == [0, 0, 20, 20]
        image = Image.open(io.BytesIO(z.read(coco["images"][0]["file_name"])))
        assert image.size == (70, 60)
        assert len(json.loads(z.read("corrections.json"))) == 1


def test_api_upload_validation_and_cross_origin_protection(store, tmp_path):
    client = TestClient(create_app(store=store, root=tmp_path))
    assert client.get("/api/frames").status_code == 200
    bad = client.post(
        "/api/import",
        files={"file": ("bad.jpg", b"bad", "image/jpeg")},
        data={"capture_group": "new", "split": "development"},
    )
    assert bad.status_code == 422
    response = client.put(
        "/api/frames/f1",
        json={"revision": 0, "annotations": []},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert store.get_frame("f1")["revision"] == 0
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), "blue").save(buffer, format="PNG")
    good = client.post(
        "/api/import",
        files={"file": ("upload.png", buffer.getvalue(), "image/png")},
        data={"capture_group": "new", "split": "development"},
    )
    assert good.status_code == 200, good.text
    imported = good.json()
    assert imported["meta"]["width"] == 40
    conflict = client.post(
        "/api/import",
        files={"file": ("another.png", buffer.getvalue(), "image/png")},
        data={"capture_group": "new", "split": "validation"},
    )
    assert conflict.status_code == 422
    image = client.get("/api/frames/" + imported["meta"]["frame_id"] + "/image")
    assert image.status_code == 200


def test_prediction_results_do_not_replace_saved_annotations(store):
    store.save("f1", [area(bbox_xyxy=[15, 15, 45, 45])], 0)
    store.add_prediction_run(
        "f1",
        {"boxes": [{"bbox_xyxy": [20, 20, 40, 40]}], "provenance": {"model": "test"}},
    )
    result = store.get_frame("f1")
    assert result["annotations"][0]["bbox_xyxy"] == [15, 15, 45, 45]
    assert result["revision"] == 1
    assert result["proposals"][0]["bbox_xyxy"] == [20, 20, 40, 40]
    assert result["original_proposals"][0]["bbox_xyxy"] == [11, 11, 31, 31]


def test_export_snapshot_stays_consistent_during_later_save(store, monkeypatch):
    from shelf_vision.editor_export import export_reviewed

    store.save("f1", [area()], 0, reviewed=True)
    snapshot = store.reviewed_snapshot

    def concurrent_save():
        records = snapshot()
        store.save("f1", [], 1)
        return records

    monkeypatch.setattr(store, "reviewed_snapshot", concurrent_save)
    result = export_reviewed(store)
    with zipfile.ZipFile(io.BytesIO(result["bytes"])) as z:
        assert len(json.loads(z.read("annotations.coco.json"))["annotations"]) == 1
        assert len(json.loads(z.read("corrections.json"))) == 1
    assert store.get_frame("f1")["status"] == "in_progress"


def test_jobs_are_model_aware_and_preserve_reviews(store):
    saved = store.save("f1", [area()], 0, reviewed=True)
    job, created = store.create_job("f1", "trained")
    assert created and job["model"] == "trained"
    same, created = store.create_job("f1", "trained")
    assert not created and same["id"] == job["id"]
    with pytest.raises(Conflict):
        store.create_job("f1", "baseline")
    assert store.get_frame("f1")["annotations"] == saved["annotations"]
    assert store.get_frame("f1")["status"] == "reviewed"


def test_api_unknown_and_unavailable_model_rejected(store, tmp_path):
    with TestClient(create_app(store=store, root=tmp_path)) as client:
        assert (
            client.post("/api/frames/f1/predict?model=not-a-model").status_code == 422
        )
        assert client.post("/api/frames/f1/predict?model=trained").status_code == 422
        models = client.get("/api/models").json()
        assert models[0]["id"] == "baseline" and models[0]["available"]
        assert not models[1]["available"]
