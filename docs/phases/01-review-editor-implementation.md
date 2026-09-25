# Phase 1: Review Editor Implementation Plan


**Goal:** Deliver the functioning local review editor against the existing pilot.

**Architecture:** FastAPI + SQLite backend, React/TypeScript SVG editor, idempotent phase 0 seed import, sequential inference worker, reviewed ROI-aware COCO ZIP export.

**Spec:** [01-review-editor-design.md](01-review-editor-design.md).

## Global constraints

Original media remains unchanged. All compute local. SKU display areas are the only detector class. Occupancy and observed identity remain separate. Human review must be explicit. Test data never enters training export. Saves must preserve immutable proposals and correction history.

## File map

- `src/shelf_vision/editor_store.py`: SQLite schema, seed import, revisioned documents, correction events, export records.
- `src/shelf_vision/editor_api.py`: FastAPI routes, upload validation, safe image serving, job execution, SPA serving.
- `src/shelf_vision/editor_export.py`: reviewed ROI crops and COCO/provenance ZIP.
- `tests/test_editor.py`: real temporary SQLite/API save/import/export/conflict tests.
- `web/src/`: React app, typed API contracts, canvas geometry helpers, styles.
- `web/package.json`, lockfile, TypeScript/Vite configuration: repeatable build.
- `README.md`, `PROJECT.md`: launch instructions and milestone evidence.

## Review focus

- Save/refresh/restart and competing revisions must never silently overwrite annotations.
- Pixel/image/ROI coordinates must stay correct under zoom, drag, and resize.
- Empty branded trays must not retain an observed SKU.
- Unreviewed regions and final-test media must not leak into export.
- Queued inference must preserve current edits and show failures.

## Task 1: Persistence and API contracts

- [x] Write failing tests for idempotent seeding, save/reload, revision conflicts, invalid occupancy/SKU combinations, and edit-after-review status.
- [x] Implement SQLite transactions with revision checks and append-only save events; immutable original proposals live separately.
- [x] Add GET catalog/frames/document/image, PUT document, and POST review endpoints with validated JSON responses.
- [x] Run focused tests and inspect source/proposal preservation on saves.

## Task 2: Import, prediction, and export

- [x] Add failing tests for invalid uploads, capture-group split conflicts, cross-origin writes, ROI coordinate export, reviewed-only selection, and test exclusion.
- [x] Decode JPEG/PNG/HEIC uploads into normalized local derivatives, retain source hash and capture group, and create a full-image ROI.
- [x] Execute phase 0 inference in a serialized worker, storing run provenance and adding proposals without overwriting documents.
- [x] Implement reviewed COCO ZIP export with cropped ROI images, split manifests, rich labels, proposals, and correction events.
- [x] Run backend suite, including actual ZIP/image inspection fixtures.

## Task 3: Browser editor

- [x] Build typed React components: frame queue, image canvas, toolbar, occupancy/identity inspector, catalog, and import dialog.
- [x] Implement coordinate-safe draw/drag/resize, zoom/pan, selection, deletion, undo/redo, and proposal acceptance.
- [x] Wire explicit save, finish review, save-and-next, export, inference job progress, and revision-conflict feedback.
- [x] Retain local drafts keyed by frame and server revision; protect unsaved navigation and refresh.
- [x] Run frontend geometry tests and TypeScript production build.

## Task 4: Integrated verification and handoff

- [x] Serve built SPA from loopback FastAPI and exercise real data in the browser.
- [x] Verify import/predict/edit/save/reload/export and pointer geometry; use a disposable imported frame for mutations.
- [x] Run the complete suite and obtain an independent code review. Fix important findings with regression evidence.
- [x] Update roadmap and README with actual launch URL, commands, limitations, and unmeasured human review speed. Commit locally; do not publish or push.
