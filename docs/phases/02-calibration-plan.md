# Detector threshold calibration

Authorized 2026-09-23. Use the existing reviewed-v1 snapshot and experiment-001 checkpoint without retraining.

Before running, fix the rule: choose maximum micro F1 on development at IoU 0.50 from confidence thresholds 0.001 and 0.005 through 0.500 in steps of 0.005. Break ties by greater precision, then greater threshold. Keep image size 640, square letterbox, NMS IoU 0.50, max_det 300. Select using development images/labels only and write an immutable lock artifact before validation inference. Re-run directly at the selected threshold to check that deployment matches the cached sweep.

Report development fit and existing-validation diagnostics separately. Development was also training data, and the checkpoint was selected using validation during training. Validation was examined previously. Neither split is an independent calibration/test set, and these results cannot retroactively change experiment-001's failed fixed-threshold result. Threshold selection does not calibrate probability estimates.

Save exact checkpoint/dataset hashes, sweep, selection rule, selected threshold, predictions and locked validation report. Reject missing development labels/predictions, dataset mismatch, changed checkpoints, and existing output directories. Preserve original experiment files and registration history. Apply the locked threshold only to the optional trained candidate, with explicit calibration provenance; baseline remains default. New-visit evaluation is still required.

Verification: synthetic selection/leakage/tie regressions, frozen artifact validation, direct-inference metric parity, candidate registration hash checks, existing test suite and production build, live model metadata and annotation preservation.
