# Fixed scale comparison and video feasibility pilot

Date: 2026-09-24. The authorized bounded experiments ran locally. Neither alternative met the fixed scale gate. The video pipeline produces a reviewable tracking demo, but does not establish improved accuracy. Model registration and live annotations remain unchanged.

## Physical-tray audit

One box now means one physical tray. Numbered overlays of all six visit-002 crops were inspected before the new inference experiment. The candidate audit proposes 30 exclusions (loose cookie groups, retail candy cartons, refrigerator drinks, hanging products), flags 26 additional tray-presence/boundary cases as uncertain, and retains 183 candidates. All decisions remain agent-reviewed, pending human confirmation. No box geometry or live annotation was changed.

The scope-filtered diagnostic retains 209 labels, including the 26 uncertain cases. It is not confirmed tray ground truth. Original 239-label metrics are reported alongside it. Removing a non-tray label makes a corresponding model detection a false positive under the stricter task definition; it is not ignored background. Additional label omissions, incorrect bounds, or non-tray targets may remain.

Review board: `reports/phase-2/tray-audit-001/index.html`. Decisions are linked to original region IDs and source annotation hashes in `audit.json`; `candidate-annotations.json` is a separate candidate version. Original snapshots and reviewed editor documents remain untouched. Original visit-001 training labels also need a tray-definition audit before retraining; their earlier broader display-area policy has not been retrospectively changed.

## Fixed scale comparison

Same checkpoint, confidence 0.035, NMS IoU 0.50, square inputs, max 300. Compared 640 whole crop, 1280 whole crop, and 960-pixel tiles with 25% overlap processed at 640 plus a 640 whole-crop pass. Tile coordinates are remapped, clipped, and merged by confidence-ordered IoU NMS. No threshold tuning or retraining.

The gate was fixed before inference: recall gain >=10 percentage points, precision loss <=5 points, and median warm processing time <=3x baseline. All conditions must pass. The examined second visit is now development evidence; these are not new independent-test results.

| Input method | Original labels: TP / FP / FN | Original precision / recall | Provisional filter: TP / FP / FN | Filter precision / recall | Warm time ratio |
| --- | --- | --- | --- | --- | --- |
| Whole 640 | 19 / 27 / 220 | 41.3% / 7.9% | 11 / 35 / 198 | 23.9% / 5.3% | 1.00x |
| Whole 1280 | 10 / 72 / 229 | 12.2% / 4.2% | 8 / 74 / 201 | 9.8% / 3.8% | 2.58x |
| Tiles 960 | 30 / 155 / 209 | 16.2% / 12.6% | 21 / 164 / 188 | 11.4% / 10.0% | 7.47x |

Both alternatives failed the accuracy gate; tiling also failed the time budget. The additional detail does not solve the current detector's task/domain mismatch at the fixed operating point. This does not prove that tiling with appropriate training or different development-selected settings cannot work. It rejects these specific configurations as replacements in this experiment.

Three warm passes per method were timed, rotating method order. Final-run medians over all six preloaded ROI crops: 44.3 ms, 114.3 ms, and 331.1 ms respectively. Timing includes prediction, tile cropping, remapping, merging, and device synchronization, but excludes disk decode, model loading, and rendering. These are Mac component measurements, not end-to-end or phone FPS. Detailed all/close/wide/manual/assisted metrics are in `comparison.json`.

Evidence: `reports/phase-2/scale-002/comparison.html`, `comparison.json`, per-method predictions, and pre-inference protocol containing checkpoint, input, helper-code, and package-version provenance. Run 001 is preserved. Run 002 repeats exactly the same configurations after review requested better provenance fields; accuracy metrics are identical. No configuration search occurred between runs.

## Video pilot

Extracted 15 upright SDR frames from IMG_3523 at requested times 6.0 through 8.8 seconds, sampled at 5 fps. Fixed ROI: [0, 960, 2160, 2880]. Used the unchanged 640 detector because neither alternative passed. Estimated camera motion using sparse optical flow and a guarded partial affine fit; associated detections using one-to-one greedy IoU matching. Tracks need three actual detections for confirmation and expire after more than two missing samples. Carried boxes do not increase observation counts. This is a simple feasibility tracker, not BoT-SORT, learned temporal fusion, or SKU recognition.

The demo distinguishes raw detections, tentative tracks, confirmed observed tracks, and carried estimates. It saves each track's highest-sharpness observed crop as a candidate reference, not a guarantee of the best identity view.

Only 3 of 14 camera transitions passed the motion-fit guard. Eleven fell back to identity motion. Visual inspection shows drifting carried boxes, duplicate/fragmented tracks, and persistent detections of non-tray packaging. Seventeen track IDs reached the confirmation rule, but this is not a count of physical trays. Without persistent physical-tray IDs across frames, unique-tray recall, ID switches, and accuracy improvement are unmeasured and explicitly null.

Evidence: `reports/phase-2/video-002/index.html`, `tracking.gif`, timestamped frame manifests, `tracks.json`, `representatives/`, and `identity-review-template.json`. Run 001 is retained; run 002 adds helper provenance and explicit timing scope with the same inference/tracking settings and same outcome. Per-frame timings exclude decoding, rendering, and crop preparation; frame zero includes cold predictor initialization. They are not end-to-end throughput measurements.

## Next action

### User corrections: audit version 2

The reviewer confirmed additional non-tray groups: IMG_3524 boxes 9–11 (Mike and Ike), and IMG_3525 boxes 1–8 (Sour Patch Kids, Bottle Caps, Chips Ahoy) plus 32 and 35 (Ritz). These 13 frame-specific exclusions are recorded with original region IDs in `reports/phase-2/tray-audit-002/audit.json`. Version 1 and the original editor remain intact. The new board is `reports/phase-2/tray-audit-002/index.html`.

Version 2 has 43 exclusions, 181 retained candidates, and 15 remaining uncertain cases, for 196 retained labels. Cached scale-002 predictions were rescored without inference or threshold changes: 640 TP/FP/FN = 7/39/189; 1280 = 4/78/192; tiled = 14/171/182. Both alternatives still fail the fixed gate. These label-dependent diagnostics remain provisional while uncertain cases remain. Provenance and details are in `tray-audit-002/rescored.json`.

Human-confirm the numbered tray audit and clarify boundaries on the uncertain cases. Create a new confirmed tray dataset, audit visit-001 training examples against the same definition, and train on diverse close/oblique views. A further tracking experiment should test camera registration at a higher sample rate or use stronger feature matching, with persistent tray-ID truth on a short segment. Do not tune this pilot repeatedly and call it an independent test. Reserve a third visit for final assessment.

## Verification

Five new tests exercise tile-edge coverage, coordinate offsets/clipping, duplicate suppression, motion-aware association, one-to-one matches, actual-observation confirmation, and stale-track expiry. Full Python suite: 63 passed, with two dependency deprecation warnings. Frontend suite: 10 passed. Independent read-only review found no blocking correctness issue; its code-provenance and timing-scope findings were addressed and the experiments rerun into new output directories. Original source snapshots, live reviews, and candidate registration are unchanged.

Reproduce into fresh output paths:

```sh
YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR="$PWD/models/ultralytics-config" .venv/bin/python scripts/run_scale_experiment.py --snapshot reports/phase-2/visit-002-evaluation/run-001 --audit reports/phase-2/tray-audit-001 --out reports/phase-2/scale-003
YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR="$PWD/models/ultralytics-config" .venv/bin/python scripts/run_video_pilot.py --scale-run reports/phase-2/scale-003 --source raw_footage/IMG_3523.MOV --out reports/phase-2/video-003
```

Media and experiment artifacts remain local and excluded from Git. The audit must be restored with the other local data to reproduce the filtered metrics.


## Interactive tray review

Run the local editor and open http://127.0.0.1:8765/audit. The page starts with the 15 uncertain boxes from tray-audit-002. Select a numbered box, choose Tray, Not a tray, or Needs adjustment, and optionally add a note. Decisions save immediately; notes save with Save note or when navigating to another box. Focus box magnifies the selected region; Fit shelf restores context. All boxes makes the full audit available.

Review decisions and change history are stored in `data/editor/audit-review.sqlite`, separate from frozen annotations and the live shelf editor. The 13 explicit user exclusions start as Not a tray. Other automated suggestions remain unconfirmed. Export decisions downloads the current review and history. Needs adjustment records a follow-up request; it does not alter geometry. Apply completed decisions to a new version of the evaluation labels before rescoring.


### Interactive review completed: audit version 3

All 15 flagged cases were resolved by the reviewer: six trays and nine non-trays, with no geometry adjustments requested. Six additional retained boxes were individually confirmed. The new version at `reports/phase-2/tray-audit-003/` contains 187 retained labels and 52 exclusions. It preserves the interactive decisions and event history, source hashes, and original box geometry. The remaining 205 items carry forward their previous audit classifications; they are not represented as individually confirmed in the interactive review. Original snapshots and live editor annotations remain unchanged.

Cached predictions were rescored at the same confidence 0.035 and matching IoU 0.5:

| Method | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole image, 640 | 6 | 40 | 181 | 13.04% | 3.21% |
| Whole image, 1280 | 2 | 80 | 185 | 2.44% | 1.07% |
| 960 tiles plus whole image | 12 | 173 | 175 | 6.49% | 6.42% |

Neither alternative passes the fixed improvement gate. Tiling increases recall by 3.21 percentage points, loses 6.56 points of precision, and takes 7.47 times the baseline runtime. These are development diagnostics, not independent test results. No detector was promoted.

Next: apply the physical-tray definition to the original training annotations before another training experiment, broaden the labeled views, and reserve a third visit for final assessment. Keep the unsuccessful scale experiment and the limited video pilot in the experiment record as documented evidence of model-selection decisions.
