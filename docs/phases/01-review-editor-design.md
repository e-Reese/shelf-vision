# Phase 1: Local review editor

Status: Implemented and engineering-verified. Human timed review remains pending.

## Outcome

A working local desktop browser app for the complete import, predict, correct, save, review, and export loop. Use the existing 40-product catalog and ten pilot frames on first launch. Keep originals and immutable model proposals separate from editable annotations.

## Interface

Three columns: frame queue, large image canvas, and selected-area inspector/catalog. An off-white workspace, dark photo stage, restrained green selection, and clear status labels keep attention on imagery. The header exposes import and export; the canvas action bar exposes save and review completion. No marketing screens.

The frame queue shows a thumbnail, source name, review status, and area count. Selecting a frame opens its saved draft and existing pilot review ROIs. The canvas supports fit-to-view, wheel zoom, panning, drawing a rectangle, selecting, dragging, resizing by handles, and deleting. Undo/redo applies to unsaved changes. Escape clears selection; keyboard shortcuts do not fire while typing in fields.

The inspector edits occupancy independently of product identity. Occupied regions can be known, unknown, or unresolved; empty/mixed have no single observed SKU; unclear remains unresolved. Choosing a catalog SKU sets occupied/known. Switching away clears the incompatible observed SKU. Catalog cards show thumbnail, exact known variant name, and ID. Similarity is labeled as a match score, never a probability.

A separate proposals overlay displays immutable model boxes and their top matches. Explicitly adding a proposal creates an editable candidate; running inference must never replace existing edits. Pilot agent annotations load as unreviewed editable drafts. Show review boundaries clearly and restrict area geometry to the assigned ROI.

Confirm frame saves and confirms the current frame, then advances in queue order. On the final frame it reports completion or the number of earlier unconfirmed frames. Frame navigation and import save pending edits before switching, without confirming coverage. Failed saves block navigation. Clean navigation does not reopen a confirmed frame. Save remains an optional explicit action. Unsaved browser drafts are retained locally with their base server revision; browser close/reload retains an unload warning. A clear saved/error state communicates failures. A revision conflict preserves edits and requires reload/reconciliation rather than overwriting a newer save.

## Scope and persistence

Use SQLite under `data/editor/` for frames, mutable documents, original proposals, inference jobs, and append-only correction events. Seed existing pilot artifacts idempotently on first use; restarting must not reset edits. Store source identity, image dimensions, capture group, split, and ROIs on each frame. Preserve raw image bytes and normalized derivatives on import. Serve files only through ID-based routes resolved from the database.

Document saves are atomic and use optimistic revision checks. Each save records before/after documents, changed annotation IDs, action, timestamp, and revision. Finishing review is an explicit human action. Subsequent edits reopen the review. Inference jobs run sequentially in a background worker and report queued/running/completed/failed states without freezing the editor.

Imported images require a capture-group name and split assignment. Enforce one split per capture group for imported media. Existing same-visit pilot videos retain their explicitly documented development/validation allocation; do not retroactively relabel them as independent visits. Final-test images are labeled as such and excluded from training exports.

## Export

Export a ZIP containing COCO display-area annotations, cropped reviewed-ROI images, catalog metadata, split/source provenance, reviewed rich annotations, original model proposals, and correction events. Only explicitly human-reviewed frames are included; final-test data is excluded. Crop each reviewed ROI and translate boxes to crop coordinates, so unlabeled areas outside the ROI never become negative training examples. Use one COCO category, display_area; retain SKU and occupancy as extra annotation metadata. Mark ambiguous/mixed/empty cases faithfully, not as identified present products.

## Architecture

FastAPI serves JSON APIs and the built React/TypeScript SPA from one loopback origin. SQLite transactions protect saves and event history. React renders SVG annotations in original image coordinates, avoiding a canvas-library dependency for these simple rectangles. Native pointer events handle editing, and all geometry is bounded in image/ROI coordinates. Vite builds static frontend assets.

The prediction adapter reuses the phase 0 pipeline in a per-job directory with a single-frame manifest and a copied catalog. This first version reloads models per job; its startup overhead is visible. Keep a future persistent model worker out of this release. No cloud services or inference APIs.

Serve on `127.0.0.1` by default. Reject cross-origin mutations, cap uploads, validate decoded images and annotation values, and do not expose arbitrary filesystem routes. UI copy describes review work, not backend implementation details.

## Acceptance

- All ten pilot frames and 40 catalog products load without manual data edits.
- Import an image, run local inference, and add a proposal without replacing an existing annotation.
- Draw, move, resize, delete, assign a SKU, mark unknown/empty, and undo/redo.
- Save, reload, and restart the server without losing corrections or falsely claiming review completion.
- Concurrent stale saves receive a conflict and preserve the current server document.
- Export only reviewed non-test regions with valid crop-relative COCO boxes and correction provenance.
- Run backend unit/integration tests, TypeScript build, and real-browser workflow checks, including narrow-screen usability and failed-request states where practical.
- Human review-time improvement remains unmeasured until the reviewer performs the timed comparison. A working app alone does not prove annotation speed or model accuracy.
