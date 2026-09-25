# Phase 0 baseline results

Status: Engineering pipeline operational. Phase acceptance remains open pending human label review, catalog identity completion, and a usable pre-labeling strategy.

## What ran

- Inventoried 125 originals: four portrait HDR MOV videos and 121 HEIC photos. All decoded; SHA-256 checks confirmed originals unchanged after preparation.
- Produced upright sRGB photo derivatives and ten HDR-to-SDR video frames using AVFoundation's explicit forceSDR policy.
- Reconciled 40 catalog groups, assigning all 121 photos. Product name/variant grouping is agent-reviewed; sizes/barcodes and human confirmation remain open.
- Created 90 candidate display-area labels in selected ROIs across ten frames: 52 development, 38 validation. Both splits come from the same visit.
- Ran Grounding DINO Tiny and DINOv2 Small locally on MPS. Checkpoint revisions are pinned in `configs/phase-0-baseline.json`.
- Ran a three-epoch YOLO11n training smoke test on eight development ROI crops and five validation ROI crops, 640-pixel input and batch size two. Training took 25.28 seconds. This does not establish useful trained-model quality.
- Exported the trained checkpoint to Core ML and executed it on the Mac. No iPhone measurement has been made.

## Provisional diagnostics

These numbers use agent-reviewed candidate labels, not human-verified ground truth. Ambiguous boundaries and identities require correction before model-selection claims. They are useful for identifying problems, not model accuracy claims.

| Metric | Development | Same-visit validation |
| --- | --- | --- |
| Display-area detection recall, IoU >=0.5 | 15/52 (28.8%) | 12/38 (31.6%) |
| Display-area detection precision | 15/22 (68.2%) | 12/20 (60.0%) |
| SKU top-1 on known occupied area crops | 14/40 (35.0%) | 5/27 (18.5%) |
| SKU top-3 on known occupied area crops | 22/40 (55.0%) | 5/27 (18.5%) |
| Correct localization and top-1 SKU, before rejection | 6/40 (15.0%) | 2/27 (7.4%) |

A provisional cosine threshold of 0.60, selected using development labels only, rejected all 8 development and 5 validation unknown examples, but accepted only 11/40 and 7/27 known products. This is not a useful auto-accept policy. Unknown sample sizes are tiny, and the same-visit split does not establish robustness.

A paired diagnostic on nine known regions gave 1/9 top-1 for both whole-area and manually selected package crops. This small subset contains side-facing packages and must be label-reviewed; it does not establish that package localization solves recognition. The reference gallery currently contains one front crop per product. Side-view catalog photos have not yet been incorporated.

No confirmed empty or mixed regions are in the current candidate pilot. Empty-tray behavior has schema/unit coverage, but visual/model accuracy on empty or mixed trays is not measured.

## Runtime and export evidence

The first detector invocation took 4.25 seconds. Subsequent measured full-frame detector forwards took approximately 0.74–0.85 seconds on MPS. These timings exclude model loading, image preprocessing, retrieval, and UI interaction. Later cache hits reuse those measurements and are not new timing samples.

Core ML export initially failed with `only 0-dimensional arrays can be converted to Python scalars`. The exporter requested NumPy <=2.3.5; the environment had resolved NumPy 2.5.3. Pinning 2.3.5 resolved conversion without retraining the checkpoint.

Post-NMS outputs at confidence 0.05 were empty for both runtimes after the very short training run. Matching empty outputs cannot establish parity. A separate raw-output comparison therefore used the identical 640×640 RGB input for PyTorch and Core ML, CPU execution, and all 8,400 output candidates per image on three validation ROI images.

All three raw comparisons passed the declared engineering tolerances: maximum box-coordinate difference <=1 pixel and maximum score difference <=0.001. The largest observed differences were 0.01493 pixels and 3.45e-8 in score. Native scores remained below 0.013, underscoring that this is conversion feasibility, not a usable detector.

Evidence files are local under `reports/phase-0/baseline/`, `reports/phase-0/evaluation/`, and `reports/phase-0/deployment/`. Dataset and model artifacts are deliberately excluded from Git.

## Decisions and next work

1. Keep the Mac/MPS and Core ML engineering path. The local hardware can execute the selected experiments.
2. Keep Grounding DINO as an optional proposal source, but its current whole-frame prompt often merges whole rows or misses areas. Do not make editor usability depend on it being accurate.
3. Keep DINOv2 as a measured baseline, not the accepted production matcher. Test verified side-view references and harder variant comparisons before selecting a replacement.
4. Use the forthcoming editor to verify the 40 catalog groups and correct the 90 candidate regions. Preserve every change and review status. A dedicated display-area detector is the likely first substantive training task.
5. The next capture should be an independent visit with clear views of the same rows, plus confirmed empty/mixed examples if naturally available. Do not rearrange stock merely to fabricate cases.

Proposed next gates: all 40 catalog identities confirmed before a recognition claim; every evaluation ROI reviewed before a headline accuracy number; phase 1 must reopen saved corrections without loss and show at least a 25% median review-time reduction against manual labeling on five matched-complexity images. Phase 2 should target a 15-percentage-point display-area recall gain over the reviewed baseline while losing no more than 5 points of precision. These product/experiment targets need confirmation after label correction; no target is claimed achieved.

## Verification and review

The automated suite currently covers source preservation, orientation, duplicate source handling, catalog references, annotation semantics, split integrity, embedding ranking, cache invalidation/identity, metric denominators, ROI normalization, stale dataset isolation, and identical export-check preprocessing.

An independent code reviewer found three important repeatability issues: cached frame IDs, stale generated training datasets, and native/Core ML preprocessing differences. Regression tests failed before the fixes and passed afterward. Export execution and parity assessment now have separate statuses; the raw parity script supplies the stronger numerical evidence.

## Execution rulings

- Initialized the fresh workspace on `phase-0` in place: there was no prior code branch to isolate. No remote or publication was created.
- Used native macOS frame extraction because installed FFmpeg lacks zscale. This makes media preparation Mac-specific.
- Used agent-reviewed candidates for exploratory diagnostics and smoke training, with human review explicitly pending. Phase acceptance and final accuracy claims remain open.
- Stopped model experimentation at the measured baseline rather than expanding into an unbounded model search. Human-reviewed labels and representative side views are the next useful inputs.
