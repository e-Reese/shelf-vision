# Physical-tray retraining preparation

The target is one physical tray per bounding box, including empty trays. A retail candy box, loose cookie pack, or product grouping without a tray is not a positive example. Recognition identity is a separate task.

## Original-label audit

The frozen `data/datasets/reviewed-v1` dataset was validated against its existing lock before visual inspection. All 100 annotations in 13 ROI crops were inspected with numbered overlays. The proposed decisions are saved separately in `reports/phase-2/training-tray-audit-001/`:

| Decision | Count |
| --- | ---: |
| Clear tray candidate | 33 |
| Proposed non-tray exclusion | 59 |
| Needs human tray/boundary review | 8 |

The clear tray candidates comprise 17 development and 16 validation labels. These are assistant visual judgments, not newly human-confirmed labels. Existing annotation geometry, source snapshots, and the live editor remain unchanged. The eight uncertain cases are ROI 0003 box 5, ROI 0005 box 8, ROI 0011 boxes 7 to 9, and ROI 0013 boxes 9 to 11. Several are extremely small fragments at ROI edges.

Review these at http://127.0.0.1:8765/audit?review=training. Decisions and history save in `data/editor/training-audit-review.sqlite`. The previous visit-2 review remains at `/audit` with its original saved history. All boxes can be inspected and overridden. Needs adjustment requires a geometry correction before freezing the next dataset; marking it does not redraw a box.

## Next bounded experiment

After the eight cases are resolved, freeze a new tray dataset with source hashes and decision provenance. Do not silently turn unresolved boxes into background. If a tray requires a boundary correction, resolve it or omit its whole ROI from the experiment, recording the omission. Preserve true negative crops where no trays are present.

Combine corrected visit-1 crops with the 187 retained visit-2 labels from `tray-audit-003`, using original source frames and crop coordinates. Carry forward prior audit classifications with their actual review provenance. Preserve visit-1 source-video development/validation assignments. Assign visit-2 IMG_3523 and IMG_3524 to development and IMG_3525 and IMG_3526 to validation. All frames from a source video must remain in the same partition. The latter two videos have assisted annotations, which must be reported as a limitation. This is development validation on already inspected footage, not a fresh independent test.

Use the original pretrained YOLO11n initialization and the existing 60-epoch settings in `configs/phase-2-training.json` to isolate the label and data changes. Do not resume the model trained on broader display-area labels. Record a separate protocol before starting; the original experiment runner enforces confidence 0.25 and must not be invoked unchanged for this comparison.

Compare the old and new checkpoints on exactly the same corrected validation crops at locked confidence 0.035, NMS IoU 0.5, and matching IoU 0.5. Report TP/FP/FN, precision, recall, F1, results by visit and view, and false positives on negative crops. Preserve a single run budget. A useful development improvement requires at least 10 percentage points more recall, at most 5 points less precision, and no more than 3 times the baseline inference time under the same timing protocol. This is not automatic deployment approval or evidence of new-store generalization.

Reserve visit 3 for final assessment. Additional adjacent video frames require matching reviewed labels; do not inherit boxes unchanged across moving-camera frames. Video fusion remains a separate experiment because the first pilot did not establish reliable camera motion or persistent tray identities.

## Current readiness

All eight flagged cases are resolved: two trays and six non-trays. No geometry corrections were requested. The corrected visit-1 version retains 35 labels and excludes 65, with 92 assistant classifications carried forward. Decisions are archived at `reports/phase-2/training-tray-audit-002/`. The combined `data/datasets/trays-v2` dataset has 88 training trays in 12 crops (six negative crops) and 134 validation trays in seven crops (one negative crop). The protocol and one-run experiment are at `reports/phase-2/tray-experiment-002/`; see the results document for final outcome.
