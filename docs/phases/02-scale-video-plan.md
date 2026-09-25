# Tray scale and video pilot

Protocol recorded 2026-09-24. One box means one physical tray. Preserve live labels, model registrations, and frozen experiments.

## Fixed scope

1. Audit the six visit-002 crops against the physical-tray rule. Save per-label decisions and a separate candidate label version. Agent judgments remain provisional until human confirmation. This examined visit is now development evidence; original test snapshots remain historical and untouched.
2. Compare the unchanged trained checkpoint at 640, 1280, and overlapping 960-pixel tiles processed at 640 with 25% overlap. Tiled mode includes a 640 whole-crop pass for large trays. Merge remapped boxes using class-agnostic IoU NMS at 0.50, confidence 0.035, max 300. Warm each method and time three complete passes; report median full-dataset time. Choose only if recall gains >=0.10, precision decreases <=0.05, and latency <=3x the 640 baseline. Report close and wide groups, all original labels and provisional tray-only labels. No threshold sweep, retraining, or deployment promotion.
3. Short video pilot: a fixed three-second segment of IMG_3523 at 5 fps. Use the selected method, or 640 if none qualifies. Link detections with camera-motion compensation and spatial overlap, allow short gaps, and render track IDs with observed versus carried boxes distinguished. Save track histories and representative crops. This is a feasibility demo until persistent-ID truth is human-reviewed; do not report unmeasured accuracy improvement. Future comparison requires unique-tray recall, false/duplicate tracks, ID switches, and time to detection.

## Implementation and checks

- [x] Audit: numbered overlays, explicit exclusion/uncertainty reasons, unchanged source hashes. Human confirmation remains pending.
- [x] Scale utilities: test tile edge coverage, invalid inputs, offset/clipping, and duplicate suppression before implementation.
- [x] Fixed runner: immutable config/input hashes before inference, deterministic prediction matching, warm timing, all metrics and overlays.
- [x] Video utility: test motion translation, one-to-one association, stale-track expiry, and confirmation based on actual observations. Extract upright SDR frames with existing AVFoundation helper.
- [x] Run the experiments, inspect visuals, document limitations and results.
- [x] Run relevant and full checks; obtain one final independent code review and resolve material findings.

## Decisions and limits

Do not treat default Unknown labels as verified out-of-catalog evidence. Keep original editor documents untouched. Video frames from one pass are correlated. Camera motion compensation can fail on parallax or repetitive texture; distinguish retained boxes from fresh detections. Performance on this development visit cannot become a new untouched test claim.

Execution ruling: the audit identified unconfirmed tray boundaries, so the filtered comparison is diagnostic only and no deployment promotion is permitted. Original metrics are retained as a sensitivity comparison. Original training-label audit and persistent-ID truth need human input before retraining and video accuracy claims. Results: [scale and video findings](02-scale-video-results.md).
