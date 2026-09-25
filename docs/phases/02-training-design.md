# Phase 2: First reviewed-data training experiment

Status: Historical training design, recorded 2026-09-23.

## Outcome and scope

Freeze the confirmed editor dataset, measure the fixed baseline against corrected labels, fine-tune a compact one-class detector locally, compare validation results, and expose the trained detector as an optional editor proposal source. Preserve all user corrections and original baseline runs. Recognition remains DINOv2 catalog retrieval; the trained model predicts display areas, including known and unknown products. Do not infer occupancy or identity from a detector box.

## Data contract

Snapshot all currently confirmed non-test frames using one SQLite read transaction. Fail if any non-test frame is not confirmed. Save the exact editor export ZIP, its SHA-256, original-source revisions, a generated dataset manifest, and content hashes. Never overwrite an existing dataset or experiment directory. The frozen export is independent of future editor edits.

Use only the reviewed ROI crops. Each COCO crop becomes a training/evaluation image with one class, display_area. Keep empty/mixed/unclear annotations if present because the detection target is the display area itself. Preserve occupancy and identity for diagnostics. No test data or external images enter training.

Development maps to train, validation maps to val. Enforce source-video/source-image isolation across splits and preserve original source identifiers. Report shared capture sessions explicitly: this pilot has one visit across development and validation. It is useful for engineering and model selection but not independent generalization evidence. Require both nonempty splits and at least one labeled training area. Validate annotations, unique IDs, crop bounds, split membership, and image hashes before compute.

## Experiment fixed before execution

- Initialization: existing local `yolo11n.pt`, record SHA-256. Do not initialize from phase 0 smoke-test weights.
- One class: display_area. Known and unknown are equally valid positive examples.
- Image size 640; batch 4; 60 maximum epochs; patience 15; seed 0; workers 0; MPS when available, otherwise CPU.
- Optimizer AdamW, initial learning rate 0.001, weight decay 0.0005, warmup 3 epochs, cosine LR. All layers trainable. AMP off on this local experiment.
- Modest scale/translation/color augmentation; no mosaic, mixup, cutmix, flips, rotation, or perspective. Preserve recognizable package geometry.
- Keep Ultralytics best checkpoint selected by validation fitness; report that selection uses validation. Store actual resolved arguments and per-epoch results.
- Detection operating point: confidence >= 0.25, one-to-one confidence-ordered matching at IoU >= 0.50. Report TP/FP/FN, precision, recall, F1, per-frame results, and known/unknown recall. Also retain native trainer mAP50 and mAP50-95 as supplementary results, not interchangeable with fixed-threshold metrics.
- Proposed promotion criterion: validation F1 improves by at least 0.10 absolute over the baseline, recall does not decrease, and precision is at least 0.50. Evaluate once for this initial configuration. Failed criteria are a valid result; do not silently sweep configurations until one passes.

## Fair baseline and diagnostics

Run the pinned Grounding DINO Tiny + DINOv2 baseline on exactly the frozen ROI images, with the existing prompt and thresholds. Training and editor candidate inference use the same ROI framing. Old whole-frame baseline results remain historical and must not be mixed into this comparison. Match predictions by image ID, never array order. Save predictions, elapsed times, and baseline recognition top-1/top-3 on corrected known-area crops. Select any retrieval rejection threshold on development only, then report validation unknown false acceptance. No recognition improvement is claimed from detector training.

Produce machine-readable comparison metrics and an HTML visual report with truth and predictions for each validation crop. Runtime includes clearly separated loading and per-image inference; it is not phone latency. Accuracy claims include image/object denominators and the same-visit warning.

## Editor integration

Keep the current baseline as the default. Add an explicit proposal-source selector for baseline vs trained display-area detector. The trained source is available only when a local model registration exists, with checkpoint hash and dataset/run metadata. Expose a candidate even if it misses the promotion gate, labeled experimental; do not quietly replace the default.

Run trained inference on each frame's review ROIs, translate crop boxes back into full-image coordinates, and retain ROI association so a proposal is attached to the correct review region. Trained proposals have detection confidence but no SKU match until a separate retrieval pass is implemented. UI copy must state this. Background inference appends a provenance-bearing run and never edits user annotations or confirmation status. Reject overlapping active jobs for the same frame if a different model was requested; do not silently return the wrong model's job.

## Acceptance and limits

A repeatable freeze/train/evaluate command path, a real locally trained checkpoint, corrected-label baseline and validation comparison, visual error report, and successful editor candidate run constitute engineering completion. Improvement only counts if the fixed promotion criterion passes. Test preprocessing, leakage guards, coordinate remapping, metric matching, and API model selection. Perform an independent final code review and fix important findings.

No new editor layout system, segmentation, shelf comparison, phone deployment, multi-seed study, or hyperparameter search in this phase. the reviewer's next independent capture visit is required before final performance claims.
