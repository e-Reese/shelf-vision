# Physical-tray retraining results

The corrected-label YOLO11n run substantially reduced false detections, but did not increase overall tray recall at the fixed operating point. It fails the predeclared improvement gate. No model was promoted or installed as the editor default.

## Data and training

The reviewer resolved the eight uncertain original-label cases as two trays and six non-trays. The resulting visit-1 version has 35 retained labels and 65 exclusions. The other 92 decisions carry forward assistant visual classifications; they are not represented as individually human-confirmed. Visit 2 contributes 187 retained tray labels with the previously recorded mixed review provenance. Original images, box coordinates, and prior snapshots remain unchanged.

The immutable dataset is `data/datasets/trays-v2`. Source videos are disjoint across partitions:

| Source | Training crops / trays | Validation crops / trays |
| --- | ---: | ---: |
| Visit 1, original assignments | 8 / 17 | 5 / 18 |
| Visit 2 | 4 / 71 | 2 / 116 |
| Total | 12 / 88 | 7 / 134 |

Visit-2 IMG_3523 and IMG_3524 are training sources; IMG_3525 and IMG_3526 are validation sources. Six training crops and one validation crop contain no trays. Negative crops remain in the dataset. Every normalized YOLO label was independently converted back to pixels and matched its frozen source box within 0.001 pixel.

One YOLO11n run started from the original pretrained `yolo11n.pt`, using the existing settings: 60 epochs, seed 0, image size 640, batch 4, AdamW, and the existing augmentation policy. Training completed on Apple MPS in 95.87 seconds. The trainer selected `train/weights/best.pt` using its normal validation fitness; it was not selected by trying multiple runs or thresholds. PyTorch warned that some MPS operations are nondeterministic despite the deterministic setting, so exact bitwise reproduction is not guaranteed.

## Fixed paired comparison

Both the original display-area checkpoint and the new tray checkpoint saw the same seven corrected validation crops at confidence 0.035, NMS IoU 0.5, matching IoU 0.5, square 640 inference, and maximum 300 detections.

| Model | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original checkpoint | 13 | 45 | 121 | 22.41% | 9.70% | 13.54% |
| Corrected-tray checkpoint | 13 | 7 | 121 | 65.00% | 9.70% | 16.88% |

False detections fell by 84.4%. Both models found 13 trays, but not the same mix of trays:

| Validation subset | Original TP / FP / FN | New TP / FP / FN |
| --- | ---: | ---: |
| Visit 1, closer and oblique views | 13 / 30 / 5 | 9 / 5 / 9 |
| Visit 2, wide views with assisted labels | 0 / 15 / 116 | 4 / 2 / 112 |
| Tray-free crop (subset of visit 1) | 0 / 7 / 0 | 0 / 0 / 0 |

Visit-1 recall declined from 72.22% to 50%; visit-2 recall rose from zero to 3.45%. These changes explain why overall recall remained flat. The wide-view problem is largely unresolved. The inspected overlays show cleaner close-view predictions but very few detections on the densely stocked wide views.

Three alternating warm passes over preloaded validation images took median 52.80 ms for the old model and 51.94 ms for the new model for all seven crops. This timing excludes image decoding and model loading, includes device synchronization, and is not an iPhone or end-to-end video benchmark.

The gate required at least 10 percentage points more recall, no more than 5 points lost precision, and at most 3 times the original inference time. Recall gain was zero; the other two conditions passed. The overall gate failed.

## Interpretation and next decision

This run supports a narrower conclusion: aligning training labels to the physical-tray target and adding these reviewed views reduced false detections at the locked threshold. Because labels and data both changed, the experiment does not isolate their separate effects. It does not demonstrate adequate tray coverage or cross-visit generalization.

The validation material was already inspected, shares the same store and shelves, and the visit-2 validation annotations were proposal-assisted. Some labels retain assistant audit judgments and existing imperfect geometry. Treat all results as development diagnostics. Visit 3 remains reserved for a genuinely fresh assessment.

Keep this completed bounded experiment in the experiment record, including its failed gate. Before another training run, collect and label more varied medium and close views of the trays missed in wide shots, with consistent physical-tray boundaries and broader source-video coverage. Any new scale, threshold, or temporal-fusion experiment needs a separate declared budget and protocol. The earlier video pilot still lacks reliable motion registration and persistent tray-ID truth; this training run does not change that finding.

The [machine-readable comparison](../results/tray-comparison.json) is included in this repository. Image overlays and checkpoints remain local.

## Local artifacts and reproduction

- Final original-label audit: `reports/phase-2/training-tray-audit-002/`.
- Frozen training data and hashes: `data/datasets/trays-v2/lock.json`.
- Protocol, checkpoint, paired predictions, metrics, overlays: `reports/phase-2/tray-experiment-002/`.
- Visual comparison: `reports/phase-2/tray-experiment-002/comparison.html`.
- Training log: `reports/phase-2/tray-experiment-002.log`.
- Runner: `scripts/run_tray_retraining.py`.

The runner prepares the dataset with `prepare`, then consumes one run budget with `run`. Existing output paths are deliberately protected; reproduce into a new versioned directory after restoring the hashed local source artifacts. Do not delete the current experiment to rerun it. Media and model artifacts remain local and excluded from Git.
