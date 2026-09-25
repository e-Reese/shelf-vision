# Phase 2: First reviewed-data experiment

Date: 2026-09-23. Engineering loop complete. **The trained candidate did not pass the predeclared replacement criterion.** Original baseline remains the default.

## What ran

The reviewer confirmed ten frames containing 100 display areas: 80 known products and 20 unknowns. Frozen version `reviewed-v1` contains 13 ROI crops. Development has eight crops from six frames, with 55 areas (46 known, nine unknown). Validation has five crops from four frames, with 45 areas (34 known, 11 unknown). Source videos remain isolated across splits, but every frame comes from the same visit. Validation is not independent new-visit evidence.

Both compared detectors see exactly these frozen ROI images. The baseline uses the existing pinned Grounding DINO Tiny checkpoint and prompt. Catalog recognition remains the pinned DINOv2 Small model. The candidate starts from the existing pretrained `yolo11n.pt`, not the earlier smoke-test weights, and fine-tunes all trainable layers for one class: display_area. Unknown products remain positive detection examples.

The fixed configuration is [phase-2-training.json](../../configs/phase-2-training.json). All 60 epochs completed on MPS in **83.55 seconds**. The best validation fitness in the epoch log occurs at epoch 50; this Ultralytics version uses mAP50-95 for fitness. The selected checkpoint is approximately **5.46 MB**. Native trainer metrics at that epoch are mAP50 **0.6004** and mAP50-95 **0.39625**. These ranking metrics are not precision/recall at the declared operating threshold.

Environment: PyTorch 2.14.0, Ultralytics 8.4.160, Transformers 4.57.6, NumPy 2.3.5. Seed 0, batch 4, 640 pixels, AdamW, initial LR 0.001, no mosaic or flips. The resolved trainer configuration and per-epoch metrics are saved. MPS warned that some operations lack deterministic implementations, so fixed seeds do not imply bit-identical repeat runs.

## Primary validation result

Confidence >= 0.25; one-to-one matching at IoU >= 0.50; candidate NMS IoU 0.50. These settings were fixed before training. Validation has **45 objects in five crops**.

| Detector | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grounding DINO baseline | 24 | 19 | 21 | 55.8% | 53.3% | 54.5% |
| Trained YOLO candidate | 0 | 0 | 45 | 0.0%* | 0.0% | 0.0% |

*With no predictions, this report defines precision as zero for operational summaries; it is not an observed false-positive rate.

The promotion rule required at least +0.10 F1, no recall regression, and precision >=0.50. The candidate failed all three conditions. Its development result was seven correct detections with zero extras and 48 misses at the same threshold. Training completion is not evidence of useful default operating behavior.

## Why aggregate metrics and the operating result differ

A post-experiment diagnostic found validation maximum scores between approximately 0.07 and 0.22. All are below 0.25. On one validation crop, CPU and MPS square-input maximum scores both rounded to 0.216923. Changing to rectangular input on that crop still left the maximum below 0.25. The evidence points to low confidence, not a missing checkpoint or an MPS-only inference mismatch.

At confidence 0.05, the exploratory validation diagnostic produced 24 TP, five FP, and 21 FN: precision 82.8%, recall 53.3%, F1 64.9%. **This is post-hoc diagnostic evidence, not a passed promotion result.** The threshold was examined after validation results were available. Do not replace the primary result or tune repeatedly against these frames and call them an untouched test. The existing candidate registration retains confidence 0.25.

The next experiment should choose a calibration policy on development data before evaluation, and use new capture data for independent assessment. More labeled images and diverse views may also improve confidence and recall; this run does not prove which intervention will help most.

## Recognition remains a separate limitation

DINOv2 recognition on human-confirmed known display-area crops:

| Split | Top-1 | Top-3 | Known accepted at 0.60 | Unknown falsely accepted at 0.60 |
| --- | ---: | ---: | ---: | ---: |
| Development | 14/46 (30.4%) | 23/46 (50.0%) | 11/46 (23.9%) | 0/9 |
| Validation | 5/34 (14.7%) | 6/34 (17.6%) | 7/34 (20.6%) | 0/11 |

The rejection threshold 0.60 was selected using development only: maximize known acceptance while development unknown false acceptance is <=10%. Zero false acceptance on 11 unknown validation areas is a small sample, not a general guarantee. Display-area crops can contain tray artwork, multiple packages, occlusion, and substantial context; recognition needs its own experiment. The trained editor source currently supplies boxes only, with manual identity assignment.

## Runtime and product integration

Across the 13 ROI images, median warm candidate prediction wall time was approximately **24.8 ms**. Baseline detector inference median was approximately **721 ms**; its complete baseline/retrieval run took 16.62 seconds. These timing scopes differ: the baseline per-image detector measurement excludes image preparation while candidate wall time includes its prediction call. Treat them as component observations, not a controlled end-to-end speedup ratio or phone FPS.

The editor now offers Original baseline + SKU matches and Trained detector (pilot). Baseline remains default. The trained option is explicitly marked experimental and failed-validation; its latest proposal provenance is displayed separately from the selected source. It predicts each ROI, translates coordinates into the original image, and preserves the ROI association when a proposal is accepted.

An actual candidate job in an isolated editor database completed in approximately 0.87 seconds, including loading and two ROI predictions, and produced one proposal on the first pilot frame. Saved annotations, revision, and confirmation status remained identical. Browser model selection and completion worked with no console errors. All 100 live user annotations and ten confirmed statuses were verified unchanged.

## Provenance and reproduction

- Dataset ID: `bf38e3d9a988d74012c6bbf33eed4ff948605552931e6b15600704661c8cd717`.
- Checkpoint SHA-256: `163b9a97dffdc69e0f82d2455058bb9bdb53fb7b5fb7eaa87ea8c1d12a00f333`.
- Dataset: `data/datasets/reviewed-v1/dataset.json`, exact `reviewed.zip`, hashed crop/reference images, annotations, and YOLO labels.
- Run: `reports/phase-2/experiment-001/experiment.json`, `comparison.json`, `trained-predictions.json`, `baseline/predictions.json`, `train/args.yaml`, and `train/results.csv`.
- Checkpoint: `reports/phase-2/experiment-001/train/weights/best.pt`.
- Visual comparison: `reports/phase-2/experiment-001/validation.html`.
- Confidence diagnostic: `reports/phase-2/experiment-001/confidence-diagnostic.json`; repeatable script is `scripts/check_phase2_confidence.py`.
- Editor registration: `models/phase-2-candidate.json`; registered file hash is checked before inference.

Artifacts with media or model weights remain local and outside Git. See [README](../../README.md#reviewed-data-training) for commands. Frozen snapshots use absolute paths and must stay in their original location; re-freeze from reviewed data into a new directory instead of silently copying one and training against the wrong files. Existing dataset and experiment directories cannot be overwritten.

## Verification and review

51 Python tests and ten frontend tests pass; TypeScript production build passes. Coverage includes source split leakage, unknown positives, ROI labels, missing/duplicate/reordered predictions, empty detections, fixed operating point, model-aware job conflicts, checksum rejection, coordinate mapping, and preserved reviews.

Independent review found four important integrity issues. Regression tests reproduced them, then verified fixes: reject unexpected input files; verify consumer paths stay in the checked snapshot; reject malformed/empty/out-of-image ROIs and dropped annotations; reject unsupported confidence/matching settings rather than misreporting metrics. Generated label caches are cleared before training. The actual experiment snapshot passes the strengthened checks; it had no extra input images, used its original path, valid ROIs, and the supported fixed settings.

Engineering completion is separate from accuracy acceptance. The next decision is a predeclared confidence-calibration and additional-capture experiment. Shelf comparison and physical iPhone deployment remain future phases.
