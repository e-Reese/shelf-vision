# Visit 2: locked detector evaluation

Date: 2026-09-24. All six frames confirmed by the reviewer; evaluation completed without retraining or threshold selection. These are measured results against the submitted labels. A target-definition audit is needed before using aggregate numbers as a model accuracy claim.

## Frozen inputs and settings

239 annotations across six ROI crops from a separate store visit: 19 areas in three close views, 220 in three wide views. Four frames (84 annotations) were confirmed before baseline assistance; the last two (155 annotations) received Grounding DINO proposals before confirmation. All remain in the test split and outside normal training export.

The snapshot contains a consistent SQLite backup, reviewed documents and revisions, copied ROI images and catalog references, hashes, and the predeclared protocol. Model predictions use the same frozen ROI crops. Grounding DINO uses its pinned checkpoint, existing prompt, and box/text thresholds of 0.25. YOLO uses the original trained checkpoint, locked confidence 0.035, image size 640, square letterbox, NMS IoU 0.50, and maximum 300 detections. Both use confidence-ordered one-to-one matching at IoU >=0.50. No test-driven tuning or model promotion was performed.

## Results

| Group | Model | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All six, 239 labels | Grounding DINO | 29 | 42 | 210 | 40.8% | 12.1% | 18.7% |
| All six, 239 labels | YOLO | 19 | 27 | 220 | 41.3% | 7.9% | 13.3% |
| Four manual, 84 labels | Grounding DINO | 16 | 27 | 68 | 37.2% | 19.0% | 25.2% |
| Four manual, 84 labels | YOLO | 15 | 16 | 69 | 48.4% | 17.9% | 26.1% |
| Two assisted, 155 labels | Grounding DINO | 13 | 15 | 142 | 46.4% | 8.4% | 14.2% |
| Two assisted, 155 labels | YOLO | 4 | 11 | 151 | 26.7% | 2.6% | 4.7% |
| Three close, 19 labels | Grounding DINO | 9 | 23 | 10 | 28.1% | 47.4% | 35.3% |
| Three close, 19 labels | YOLO | 13 | 13 | 6 | 50.0% | 68.4% | 57.8% |
| Three wide, 220 labels | Grounding DINO | 20 | 19 | 200 | 51.3% | 9.1% | 15.4% |
| Three wide, 220 labels | YOLO | 6 | 14 | 214 | 30.0% | 2.7% | 5.0% |

Groups overlap; do not add their totals. These are views from one visit, not independent store/session replicates. Annotation assistance may bias baseline comparison. The small manual-group F1 difference does not establish a general improvement.

## Visual findings and limits

- YOLO produces useful close-view localizations but sometimes merges large sections of shelves in wide views. Both detectors miss many wide-view targets.
- At YOLO's 640-pixel letterboxed input, the median shortest annotated box side is approximately 184.1 pixels for close views and 34.7 pixels for wide views. Scale is a plausible contributor; this experiment does not isolate it from viewpoint, occlusion, or target-definition differences.
- IMG_3525 labels include individual refrigerator drinks. The original task is SKU display areas or trays, with contiguous product groups where no tray boundary exists. Review whether those boxes and other dense annotations use the same detection unit as the original training labels. Do not silently remove them or reframe the measured result after seeing predictions.
- Unknown identity became a drawing default and was applied in bulk. Identity-stratified metrics are retained in the machine report but are not evidence of verified out-of-catalog recognition. Identity does not affect the detection totals above.
- Neither SKU-recognition accuracy nor phone performance is evaluated here. Baseline default and registered model settings remain unchanged.

## Next experiment

First audit the detection unit on representative wide-view labels and write any clarification explicitly. Preserve this snapshot and score as submitted. If labels need corrections, create a separately versioned snapshot and report the reason and both versions. Then use development data to test a small-object strategy, such as tiled inference or higher resolution, and collect diverse labeled wide views for training. This visit has now been examined and can support diagnosis; reserve another visit for a future untouched final test.

## Evidence and verification

Local directory: `reports/phase-2/visit-002-evaluation/run-001/`.

- `comparison.html`: side-by-side labels and predictions for every crop.
- `comparison.json`: all groups, per-frame metrics, settings references, and caveats.
- `snapshot-lock.json`, `protocol.json`, `sources.json`, `editor-snapshot.sqlite`: frozen inputs and provenance.
- `baseline/predictions.json`, `trained-predictions.json`: actual outputs.
- `compare.py`: candidate inference, shared metric computation, and visual report generation.

Verified model checkpoint and threshold-lock hashes, confirmed status, image hashes, ROI membership, schema consistency, unique annotation IDs, and durable snapshot hashes before and after candidate evaluation. The original snapshot lock inadvertently included SQLite WAL/SHM sidecars, which disappeared when the connection closed. Verification excludes only these transient files; the main database and all durable inputs match the original pre-inference lock. The original lock is preserved and the exception is recorded in `comparison.json`.

26 existing evaluation, training, and calibration tests pass. Experiment execution is read-only against the live editor and does not modify labels or model registration.
