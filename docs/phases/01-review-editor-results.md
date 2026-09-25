# Phase 1: Review editor verification

Date: 2026-09-22. Engineering implementation complete. Human review and timing acceptance remain open.

## Delivered

The local React/TypeScript editor and FastAPI/SQLite service support image import, sequential local model inference, editable display-area rectangles, catalog assignment, unknown/empty states, proposal overlays and explicit acceptance, undo/redo, browser drafts, revisioned saves, explicit review completion, and reviewed-region COCO ZIP export.

The interface opens the ten pilot frames, 90 candidate annotations, and 40 provisional catalog products. Pilot annotations remain unreviewed. Original predictions are immutable; later model runs are appended separately from corrections. Export freezes eligibility, annotations, and audit records under one SQLite read transaction.

Launch instructions: [README](../../README.md#launch-the-editor). Design: [editor design](01-review-editor-design.md). Task record: [implementation plan](01-review-editor-implementation.md).

## Verification evidence

| Check | Observed result |
| --- | --- |
| Python suite | 33 passed; two dependency deprecation warnings from FastAPI/Starlette test support |
| Frontend tests | Six passed: bounded geometry, camera fitting, undo draft cleanup, conflict draft archival |
| TypeScript and production build | Passed; approximately 260 kB JavaScript before gzip |
| Python undefined/unused-name check | Passed for editor modules and editor tests |
| Import | A 2160 × 3840 PNG copy of pilot frame 1 imported through the browser; original bytes retained |
| Local inference | Grounding DINO Tiny and DINOv2 Small produced five proposals; queued-to-complete job elapsed approximately 6.6 seconds, including startup |
| Editing | Draw, move, corner resize, undo, and redo produced expected image-coordinate rectangles; catalog click assigned Oreo Original |
| Occupancy and identity | Empty changed identity to not-applicable and cleared SKU; unknown cleared SKU; undo restored prior state |
| Delete | Selected area removed, undo restored it; no draft remained after undo to the saved state |
| Save/reload | Browser reload preserved rectangle `[750, 2150, 1250, 2650]` and assigned `sku-0017` |
| Export | Explicit review enabled a ZIP with one image and one annotation; COCO bbox `[750, 2150, 500, 500]` verified by reading the downloaded archive |
| Revision conflict | A competing save caused HTTP 409; corrections remained in browser, then archived on reopen with a downloadable draft backup |
| Review eligibility | Automated fixtures exclude unreviewed and final-test frames; later saves reopen review |
| Concurrent export | Regression test performs a save after export snapshot and verifies the ZIP retains the reviewed revision and its corresponding history |
| Original media | All 125 SHA-256 checksums match the original inventory |
| Browser console | No console errors on the final clean reload |
| Responsive layout | Screenshots inspected at 1440, 768, and 375 pixel widths; 375 pixel viewport had no document horizontal overflow |

Browser interaction used the gstack browse skill. Native clicks, selects, upload, reload, dialogs, and downloads exercised the workflow. Draw/move/resize were additionally checked with DOM pointer-event simulation, temporarily stubbing pointer capture in the test page. This verifies handlers and coordinate mapping but does not replace a human check of trackpad drag feel and real pointer capture. Geometry tests cover clipping and size preservation independently.

Evidence is local and excluded from Git: `reports/phase-1/editor-desktop.png`, `editor-tablet.png`, `editor-mobile.png`, `pointer-check.js`, and `qa-reviewed.zip`. The disposable QA frame and its database records were removed after verification; the original pilot has no synthetic human-review approvals.

## Independent review and fixes

The independent reviewer identified three important issues. All were fixed and checked:

1. Export could race a save and include a reopened draft. Export now reads consistent SQLite snapshot records before image generation.
2. Reopening after a conflict could allow the next edit to overwrite the preserved draft. Stale revisions are archived under unique keys before replacement, and Draft backup downloads them for manual reconciliation.
3. Undo to saved state left a stale local draft. Returning to the saved document now removes the current-revision draft.

## Remaining acceptance and limitations

The reviewer should complete and time a full frame review, verify provisional product identities, and check pointer interaction on the Mac trackpad. No annotation-speed improvement or model-accuracy improvement is claimed. The current baseline still requires substantial correction.

New imports use a full-image review region; editing region boundaries is deferred. Models reload for each job. The editor retains the original one-visit pilot development/validation allocation, which is not independent final evaluation. A new capture session is required for final testing.

The app is loopback-only, intended for one operator, and has no phone camera integration. It does not yet train from editor exports, compare shelf layouts, or deploy inference to the iPhone. These remain later phases. Preserve development/validation split metadata when building phase 2 training data.

## Follow-up: confirmation and navigation, 2026-09-23

User review found that separate save and review completion were unclear. The primary action is now Confirm frame: save, confirm, and advance without a second confirmation dialog. Switching frames and importing automatically persist dirty edits before navigation; save errors leave the current frame and draft intact. Next frame also autosaves. Clean saves/navigation preserve confirmation. Final-frame confirmation reports completion or remaining earlier frames.

Ten frontend tests and the production build pass. Browser checks against an isolated SQLite copy verified navigation autosave, confirm-and-advance, blocked navigation on a simulated save failure, and final-frame completion. Live user records were untouched by QA; all ten live frames were confirmed when checked. Browser reload/close retains the unsaved-change warning and local draft recovery.
