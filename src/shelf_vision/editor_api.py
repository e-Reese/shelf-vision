"""Loopback-only editor API and static app server."""

import hashlib
import io
import json
import os
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from .trained_detector import load_registration, predict_trained
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from .media import save_json
from .editor_store import EditorStore, Conflict, seed_pilot
from .editor_export import export_reviewed


class SaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0, strict=True)
    annotations: list[dict]


def public_document(doc):
    return {**doc, "meta": {k: v for k, v in doc["meta"].items() if k != "path"}}


def create_app(store=None, root=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    if store is None:
        catalog = json.loads((root / "data/manifests/catalog.json").read_text())[
            "products"
        ]
        store = EditorStore(root / "data/editor/editor.sqlite", catalog)
        seed_pilot(store, root)
    store.recover_jobs()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shelf-inference")

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Shelf Vision", lifespan=lifespan)
    app.state.store = store
    from .audit_review import audit_router

    app.include_router(audit_router(root))

    @app.middleware("http")
    async def local_mutations(request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.url.netloc}"
            if origin and origin.rstrip("/") != expected:
                return JSONResponse(
                    {"detail": "Cross-origin changes are not allowed."}, status_code=403
                )
        return await call_next(request)

    @app.exception_handler(Conflict)
    async def conflict_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def value_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing_handler(request, exc):
        return JSONResponse({"detail": "Not found"}, status_code=404)

    @app.get("/api/catalog")
    def catalog():
        return [
            {
                "sku_id": p["sku_id"],
                "name": p["name"],
                "size": p.get("size"),
                "status": p.get("human_review_status", "pending"),
                "thumbnail": f"/api/catalog/{p['sku_id']}/image",
            }
            for p in store.catalog
        ]

    @app.get("/api/catalog/{sku_id}/image")
    def catalog_image(sku_id: str):
        if sku_id not in {p["sku_id"] for p in store.catalog}:
            raise KeyError(sku_id)
        path = root / "data/derived/catalog" / f"{sku_id}.jpg"
        if not path.exists():
            raise KeyError(sku_id)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/frames")
    def frames():
        return store.list_frames()

    @app.get("/api/frames/{frame_id}")
    def document(frame_id: str):
        return public_document(store.get_frame(frame_id))

    @app.get("/api/frames/{frame_id}/image")
    def frame_image(frame_id: str):
        meta = store.get_frame(frame_id)["meta"]
        return FileResponse(meta["path"])

    @app.put("/api/frames/{frame_id}")
    def save(frame_id: str, body: SaveRequest):
        return public_document(store.save(frame_id, body.annotations, body.revision))

    @app.post("/api/frames/{frame_id}/review")
    def review(frame_id: str, body: SaveRequest):
        return public_document(
            store.save(frame_id, body.annotations, body.revision, reviewed=True)
        )

    @app.get("/api/frames/{frame_id}/events")
    def events(frame_id: str):
        store.get_frame(frame_id)
        return store.events(frame_id)

    @app.post("/api/import")
    async def import_image(
        file: UploadFile = File(...),
        capture_group: str = Form(...),
        split: str = Form("development"),
    ):
        group = capture_group.strip()
        if not group or len(group) > 100:
            raise ValueError("Give this capture a session name of 1–100 characters.")
        if split not in {"development", "validation", "test"}:
            raise ValueError("Invalid split")
        raw = await file.read(25 * 1024 * 1024 + 1)
        if len(raw) > 25 * 1024 * 1024:
            raise ValueError("Image exceeds the 25 MB upload limit.")
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError("Image exceeds 40 megapixels.")
                if image.format not in {"JPEG", "PNG", "HEIF", "HEIC"}:
                    raise ValueError("Use a JPEG, PNG, or HEIC image.")
                image.load()
                upright = ImageOps.exif_transpose(image)
                icc = upright.info.get("icc_profile")
                if icc:
                    upright = ImageCms.profileToProfile(
                        upright,
                        ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGB",
                    )
                rgb = upright.convert("RGB")
                clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError(
                "The file could not be decoded as a supported image."
            ) from exc
        fid = "upload-" + str(uuid.uuid4())
        directory = root / "data/editor/uploads" / fid
        directory.mkdir(parents=True)
        source = directory / "original"
        source.write_bytes(raw)
        path = directory / "image.png"
        clean.save(path)
        meta = {
            "frame_id": fid,
            "name": Path(file.filename or "Imported image").name[:200],
            "path": str(path),
            "width": clean.width,
            "height": clean.height,
            "source_id": hashlib.sha256(raw).hexdigest(),
            "capture_group": group,
            "split": split,
            "rois": [
                {
                    "roi_id": fid + "-full",
                    "bbox_xyxy": [0, 0, clean.width, clean.height],
                }
            ],
        }
        try:
            doc = store.add_frame(meta, [], [], enforce_group=True)
        except Exception:
            # Only this request's newly-created files are removed on rejected import.
            source.unlink()
            path.unlink()
            directory.rmdir()
            raise
        return public_document(doc)

    @app.get("/api/models")
    def models():
        available = True
        reason = None
        registration = None
        try:
            registration = load_registration(root)
        except (ValueError, OSError, KeyError):
            available = False
            reason = "No valid trained checkpoint registered"
        return [
            {
                "id": "baseline",
                "label": "Original baseline + SKU matches",
                "available": True,
            },
            {
                "id": "trained",
                "label": registration.get("label", "Trained detector (pilot)")
                if registration
                else "Trained detector (pilot)",
                "calibration": registration.get("calibration")
                if registration
                else None,
                "available": available,
                "reason": reason,
                "promotion_passed": registration.get("promotion", {}).get(
                    "passed", False
                )
                if registration
                else False,
                "dataset_id": registration.get("dataset_id") if registration else None,
            },
        ]

    def execute_prediction(jid, frame_id, model, registration=None):
        store.update_job(jid, "running")
        try:
            if model == "trained":
                result = predict_trained(
                    store.get_frame(frame_id)["meta"], registration
                )
                result["provenance"]["job_id"] = jid
                store.add_prediction_run(frame_id, result)
                store.update_job(jid, "completed")
                return
            from .baseline import run_baseline

            os.environ.setdefault("HF_HOME", str(root / "models/huggingface"))
            meta = store.get_frame(frame_id)["meta"]
            jobdir = root / "data/editor/jobs" / jid
            data = jobdir / "manifests"
            save_json(
                data / "frames.json",
                {
                    "schema_version": 1,
                    "frames": [{**meta, "video_id": meta["source_id"]}],
                },
            )
            save_json(
                data / "catalog.json", {"schema_version": 1, "products": store.catalog}
            )
            run_baseline(
                root / "configs/phase-0-baseline.json",
                data,
                jobdir / "results",
                gallery_dir=root / "data/derived/catalog",
            )
            result = json.loads((jobdir / "results/predictions.json").read_text())
            frame = result["frames"][0]
            store.add_prediction_run(
                frame_id,
                {
                    "boxes": frame["boxes"],
                    "provenance": {
                        "model": "baseline",
                        "detector": frame["provenance"],
                        "embedding": result["embedding_provenance"],
                        "job_id": jid,
                        "total_seconds": result["total_seconds"],
                        "device": result["device"],
                    },
                },
            )
            store.update_job(jid, "completed")
        except Exception as exc:
            store.update_job(jid, "failed", f"{type(exc).__name__}: {exc}")

    @app.post("/api/frames/{frame_id}/predict")
    def predict(frame_id: str, model: Literal["baseline", "trained"] = "baseline"):
        registration = load_registration(root) if model == "trained" else None
        job, created = store.create_job(frame_id, model)
        if created:
            executor.submit(
                execute_prediction, job["id"], frame_id, model, registration
            )
        return job

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return store.job(job_id)

    @app.get("/api/export")
    def export():
        result = export_reviewed(store)
        if not result["frame_count"]:
            raise ValueError(
                "Finish reviewing at least one non-test frame before exporting."
            )
        return Response(
            result["bytes"],
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="shelf-vision-reviewed.zip"'
            },
        )

    dist = root / "web/dist"
    if (dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/")
    def home():
        if not (dist / "index.html").exists():
            return JSONResponse(
                {"detail": "Build the frontend first: cd web && npm run build"},
                status_code=503,
            )
        return FileResponse(dist / "index.html")

    return app


def main():
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
