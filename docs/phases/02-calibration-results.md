# Development-selected detector threshold

Date: 2026-09-23. Selected confidence **0.035**, applied to the optional trained editor source. Baseline remains default. This is detection threshold selection, not probability calibration.

## Protocol and lock

Followed [the predeclared calibration plan](02-calibration-plan.md): maximize development micro F1 at matching IoU 0.50, then precision, then threshold. Grid: 0.001 plus 0.005 to 0.500 in increments of 0.005. Fixed 640 square input, NMS IoU 0.50, max_det 300; unchanged experiment-001 weights and reviewed-v1 data. Threshold selection consumed only development predictions and labels. Validation predictions were generated after writing threshold-lock.json. Direct inference at 0.035 exactly reproduced development TP/FP/FN from the cached sweep.

Development also supplied training examples, and the checkpoint was already selected using validation. These validation images were examined in prior diagnostics. Neither split is an independent calibration or test set. This follow-up does not rewrite the original experiment's failed 0.25-threshold result.

## Measurements

| Split/model | True positives | Extra boxes | Missed areas | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development, trained at 0.035 | 52 | 3 | 3 | 94.5% | 94.5% | 94.5% |
| Existing validation, baseline | 24 | 19 | 21 | 55.8% | 53.3% | 54.5% |
| Existing validation, trained at 0.035 | 30 | 13 | 15 | 69.8% | 66.7% | 68.2% |

Validation contains 45 areas across five ROI crops from four frames. The calibrated candidate's diagnostic F1 is 13.6 percentage points above the baseline. Known-area recall improves from 19/34 to 26/34; unknown-area recall decreases from 5/11 to 4/11. The overall gain does not mean every category improved.

Neighboring development thresholds have similar scores: 0.040 gives F1 94.44%, and 0.050 gives 94.34%, versus 94.55% at 0.035. Do not interpret the selected decimal as a precisely estimated universal optimum. A new capture visit should emphasize unknown products, viewpoint changes, glare, and occlusion.

No weights were retrained. Threshold sweep and direct checks took approximately 1.66 seconds after model construction on MPS. This is not end-to-end application or phone latency.

## Editor behavior and preservation

Choose **Proposals → Proposal source → Trained detector (calibrated pilot)**, then run the model. The selector displays the development-selected threshold and pending independent evaluation. Identity assignment remains manual.

An actual browser job against an isolated database returned 11 proposals on the first frame, compared with one at the original threshold. Job provenance carries confidence 0.035 and the calibration lock hash. Saved annotations and revision remained unchanged. All 100 live annotations were independently compared with the frozen export and remain identical.

The previous model registration was archived before applying the new threshold. The original failed promotion record is preserved, and baseline remains the default. New calibrated suggestions do not replace saved annotations or retrospectively relabel old proposal runs.

## Artifacts and checks

- Run directory: `reports/phase-2/calibration-001/`.
- `protocol.json`: fixed settings, grid, and input hashes.
- `development-predictions.json`, `development-sweep.json`: development-only selection evidence.
- `threshold-lock.json`: selected operating point, written before validation inference.
- Lock SHA-256: `6027f8b8852a5ac730cc75375cea95018a5a1164b56cf841a3787f016491cba3`.
- `locked-predictions.json`, `result.json`: direct inference and existing-validation diagnostics.
- `models/registrations/before-calibration-<lock-hash>.json`: exact old registration.
- `models/phase-2-candidate.json`: active optional candidate and calibration provenance.

58 Python tests and ten frontend tests pass; production build passes. New regressions cover development-only selection, absent predictions/labels, tie-breaking, mismatched checkpoints, modified locks, archived prior registration, and unchanged original promotion state.

Next: collect and label an independent visit before inspecting predictions. Evaluate this locked operating point once, including unknown-area recall. Do not tune against that new test and still describe it as untouched.
