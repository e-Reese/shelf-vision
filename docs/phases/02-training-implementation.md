# Phase 2 Training Implementation Plan


**Goal:** Produce and integrate the first compact detector trained on human-confirmed display areas, with a fair saved comparison.

**Architecture:** Versioned editor export becomes immutable ROI manifests and YOLO files. A command-line experiment runner uses the existing baseline, Ultralytics training, and shared metric helpers. An optional local model registration connects trained ROI inference to the editor's existing job workflow.

**Tech Stack:** Python 3.12, PyTorch MPS, Ultralytics, Transformers, SQLite, React/TypeScript.

**Spec:** [02-training-design.md](02-training-design.md).

## Global constraints

Original media and human reviews remain unchanged. Both models see identical frozen ROIs. Split by source; report shared visit. No final-test training. Preserve fixed configuration and the promotion criterion. Default editor source remains baseline. All compute is local; no accuracy promise.

## Review focus

- Corrupt or incomplete snapshots must fail before training and not overwrite prior runs.
- Same source in multiple splits must fail; same visit must be reported as provisional.
- Empty predictions, duplicate boxes, reordered frame arrays, and missing prediction images must not inflate metrics.
- ROI offsets and proposal association must survive crop-to-editor translation.
- Candidate model jobs must preserve annotations, carry model provenance, and never masquerade as baseline jobs.

## Task 1: Freeze and validate reviewed data

Files: `src/shelf_vision/training_data.py`, `tests/test_training.py`.
Interfaces: `freeze_dataset(store, out, catalog_dir)` returns a manifest. `validate_dataset(path)` verifies frozen manifest and files. Output includes `dataset.json`, `reviewed.zip`, `manifests/{frames,annotations,catalog}.json`, `images/{train,val}`, `labels/{train,val}`, and `data.yaml`.

- [x] Write temporary-store tests: confirmed-only, both splits, source isolation, unknown positives, valid normalized labels, no overwrite, changed-image rejection.
- [x] Run `uv run pytest tests/test_training.py -q`; expect missing-module failure first.
- [x] Implement snapshot-to-COCO extraction without arbitrary ZIP extraction; canonical IDs, exact PNG crops, content hash inventory, and strict source/annotation validation.
- [x] Freeze the actual editor database into a fresh `data/datasets/reviewed-v1` directory; inspect split counts and revision inventory.
- [x] Run tests; expect passing dataset integrity assertions. Commit the implementation and plan, excluding media.

## Task 2: Train and compare

Files: `configs/phase-2-training.json`, `src/shelf_vision/training.py`, `src/shelf_vision/training_metrics.py`, `tests/test_training.py`.
Interfaces: `run_experiment(dataset, out, config)` writes baseline/candidate predictions, trainer artifacts, comparison.json, and validation.html. `detection_metrics(frames, annotations, predictions, confidence=.25)` returns split metrics with denominators. CLI: `python -m shelf_vision.training --dataset ... --out ... --config ...`.

- [x] Add failing metric tests for duplicate predictions, missing frames, reordered predictions, zero detections, and known/unknown recall.
- [x] Implement fixed-threshold matching, actual training configuration, checkpoint/source/dataset hashing, environment records, and side-by-side visual report. Reuse `match_boxes` and recognition metrics.
- [x] Run baseline on frozen ROIs and reviewed crop identities, then execute the 60-epoch maximum training run with the predetermined early stopping rule.
- [x] Evaluate the selected checkpoint at the fixed operating point; record actual promotion result and inspect validation overlays.
- [x] Run tests and record actual measured results; no silent experiment changes.

## Task 3: Candidate proposal integration

Files: `src/shelf_vision/trained_detector.py`, `src/shelf_vision/editor_api.py`, `src/shelf_vision/editor_store.py`, `web/src/App.tsx`, `web/src/types.ts`, `tests/test_editor.py`, `tests/test_training.py`.
Interfaces: local `models/phase-2-candidate.json` registration; GET `/api/models`; POST prediction with model ID `baseline` or `trained`. `predict_trained(meta, registration)` returns full-image boxes with ROI IDs and provenance.

- [x] Write failing tests for coordinate remapping, unknown model rejection, model-aware job conflicts, and annotation preservation.
- [x] Implement checksum-verified candidate loading, ROI inference, translation, and model-specific job metadata; keep baseline API calls backward compatible.
- [x] Add a proposal-source selector and accurate detector-only copy. Accepting trained boxes starts unresolved and preserves the proposal ROI.
- [x] Run a real candidate job against an isolated editor database, inspect returned boxes and unchanged annotations, and verify browser selection/status.
- [x] Run backend suite, frontend tests, and production build.

## Task 4: Review and handoff

- [x] Obtain one independent whole-change code review per executing-plans; fix important findings with regression tests.
- [x] Write `02-training-results.md` with dataset counts, baseline/candidate metrics, timings, checkpoint selection and hashes, actual gate result, and limitations.
- [x] Update PROJECT.md and README with repeatable commands and next capture priorities. Verify original editor annotations remain unchanged.
- [x] Commit locally, leave app available, and report the results without implying new-visit or iPhone performance.
