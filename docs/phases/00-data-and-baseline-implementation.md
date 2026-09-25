# Phase 0: Data and Baseline Implementation Plan


**Goal:** Produce a verified 40-product catalog, a reviewed display-area pilot, measured local baselines, and a training/export feasibility result.

**Architecture:** A small Python command-line pipeline writes versioned local manifests and derived media, then runs model adapters against reviewed pilot data. Static contact sheets support review; no FastAPI service or React application is required in this phase. Detection, identity, and occupancy are separate records.

**Tech stack:** Python, FFmpeg/ffprobe, macOS image conversion, PyTorch, Transformers, Ultralytics, coremltools, pytest. Resolve and lock compatible versions on the Mac during execution; do not claim untested compatibility.

**Spec:** [00-data-and-baseline-design.md](00-data-and-baseline-design.md).

**Execution status:** Inline execution approved and engineering tasks implemented. Local runs completed; human review and acceptance remain open. See [results](00-baseline-results.md). Pipeline preparation lives in `scripts/prepare_pilot.py` rather than a separate `pilot.py` module; the CLI exposes review-sheet generation. Full data review is not replaced by agent labels.

## Global constraints

- Training and inference remain local. No hosted inference APIs or cloud GPU dependencies.
- Preserve `raw_footage/` originals byte-for-byte.
- One detector class: `display_area`. Do not report individual item counts.
- Occupancy is reviewed metadata in this phase, not an automated model promise.
- All four current videos are same-visit development material. Final independent testing requires another visit.
- Never treat similarity as a calibrated probability or unresolved identity as confirmed unknown.
- Download weights only when their task begins; keep media, caches, and weights out of Git.
- Keep numerical product targets unset until the pilot supplies evidence. Record them before further optimization.

## File and artifact map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.gitignore` | Reproducible environment and exclusion rules |
| `src/shelf_vision/cli.py` | CLI entry point, named `shelf-vision` |
| `src/shelf_vision/schema.py` | Manifest validation and shared record contracts |
| `src/shelf_vision/media.py` | Source inventory, checksums, oriented/color-normalized derivatives |
| `src/shelf_vision/catalog.py` | Reference grouping, crops, catalog verification |
| `src/shelf_vision/pilot.py` | Frame selection, review regions, split checks, contact sheets |
| `src/shelf_vision/baseline.py` | Detector/embedding adapters, prediction cache, timing |
| `src/shelf_vision/evaluate.py` | Metrics, exclusions, visual error report |
| `src/shelf_vision/deployment_probe.py` | Small training and Core ML feasibility checks |
| `tests/test_{schema,media,catalog,pilot,baseline,evaluate}.py` | Focused correctness tests, no model downloads in unit tests |
| `data/manifests/` | Inventory, catalog, frames, reviewed regions and annotations |
| `data/derived/`, `data/cache/`, `models/` | Regenerable local images, predictions, downloaded/trained weights |
| `reports/phase-0/` | Local contact sheets, error figures, run results |
| `docs/phases/00-baseline-results.md` | Concise durable findings and model decision |

Git is not initialized. If starting a repository, create exclusion rules before staging: `raw_footage/`, `data/`, `models/`, `reports/`, `.venv/`, `.gstack/`, and Python caches. Track scripts, configurations, tests, and text documentation only initially. Never use a blanket add before inspecting exclusions.

## Shared contracts

Store JSON manifests with `schema_version: 1`. Use source-relative paths and SHA-256 source IDs. Use upright, full-resolution derivative pixel coordinates, with exclusive maximum corners: `[x1, y1, x2, y2]`. Persist derivative dimensions, source IDs, and video timestamps in seconds.

Annotation example:

```json
{
  "region_id": "area-0001",
  "frame_id": "frame-0001",
  "review_roi_id": "roi-0001",
  "bbox_xyxy": [20, 100, 420, 350],
  "class_name": "display_area",
  "occupancy": "occupied",
  "identity_state": "known",
  "observed_sku_id": "sku-0001",
  "intended_sku_id": null,
  "truncated": false,
  "review_status": "reviewed"
}
```

Enforce: known identity requires occupied state and a valid catalog ID; unknown requires occupied state and no observed catalog ID; empty/mixed require `not_applicable` and no single observed SKU; unclear requires unresolved identity. Occupied but unreadable products can be unresolved. Intended identity never substitutes for observed identity.

Catalog entries contain stable ID, name/variant/size fields, source photo IDs, reference crop boxes, view role (`front`, `side`, `back`, `barcode`), and verification state. Missing identity evidence remains explicit. Do not force the catalog count to 40 if inspection finds a discrepancy; surface it for reconciliation.

## Review focus

1. Portrait rotation and HDR conversion: correctly oriented boxes and stable colors. Owned by task 1.
2. Reference grouping drift: extra/missing photos must not shift all later SKU assignments. Owned by task 2.
3. Tray branding without stock: never count an empty branded tray as a present product. Owned by task 3.
4. Correlated frames or stale predictions: no cross-split video leakage, and configuration changes invalidate cache. Owned by tasks 3 and 4.
5. Unknown-product evaluation: distinguish verified unknowns, unresolved labels, and missing denominator data. Owned by task 5.

## Task 1: Traceable inventory and image preparation

**Files:** environment files, `cli.py`, `schema.py`, `media.py`, `tests/test_media.py`.

**Interface:** `shelf-vision inventory --source raw_footage --out data/manifests/media.json`; `shelf-vision prepare --manifest data/manifests/media.json --out data/derived`.

- [x] Establish the Python environment and exclusion rules. Add only dependencies needed for this task. Confirm FFmpeg color-conversion filters and HEIC decoding on this machine.
- [x] Write focused tests for duplicate-content IDs, uppercase/lowercase media extensions, invalid coordinates, and unsupported/corrupt media records. Failures must report the source file and leave originals untouched.
- [x] Implement inventory using SHA-256 and ffprobe/image metadata. Record source size, dimensions, duration, rotation, frame rate, transfer characteristics, and capture grouping; keep sensitive metadata out of reports.
- [x] Convert reference images to upright RGB derivatives. Establish explicit HDR-to-SDR video conversion settings from source metadata; save the recipe rather than relying on silent default color conversion.
- [x] Verify a portrait image and sampled video frame visually at useful resolution, and verify coordinate conversion with a known asymmetric image fixture. Reject double rotation.
- [x] Run `uv run pytest tests/test_media.py -q`, execute inventory and preparation, and compare original checksums before/after. Exit artifact: all 125 sources inventoried, with decode failures explicitly listed if any.

## Task 2: Reconcile the product catalog

**Files:** `catalog.py`, `tests/test_catalog.py`, catalog records under `data/manifests/`.

**Interface:** `shelf-vision catalog-sheet --media data/manifests/media.json --out reports/phase-0/catalog.html`; `shelf-vision catalog-validate --catalog data/manifests/catalog.json`.

- [x] Generate a local HTML contact sheet containing all 121 photos, filenames, and full-resolution links. No uploads or external assets.
- [x] Draft stable product groups from visual identity and view sequence. Record uncertain groups individually; flag identities that cannot be resolved from available evidence for human review.
- [x] Assign package crops and view roles. Use front/side crops for the first reference gallery; keep back/barcode images as provenance.
- [x] Test duplicate reference assignment, nonexistent source IDs, invalid crops, and missing front views. An extra photo must be assignable without renumbering subsequent SKUs.
- [x] Validate groups and report the verified SKU count, unresolved identities, missing views, and duplicate variants. Reconcile discrepancies against the expected 40 products.
- [x] Run `uv run pytest tests/test_catalog.py -q` and the validation CLI. Exit artifact: an auditable gallery whose verified subset can be used for retrieval; unresolved SKUs are explicitly excluded until resolved.

## Task 3: Select and review the display-area pilot

**Files:** `pilot.py`, `schema.py`, `tests/test_pilot.py`, `tests/test_schema.py`; split, frame, ROI, and annotation manifests.

**Interface:** `shelf-vision pilot-sheet --frames data/manifests/frames.json --out reports/phase-0/pilot.html`; `shelf-vision annotations-validate --data data/manifests`.

- [x] Assign whole videos to development and validation after examining coverage. Save those assignments before selecting approximately 10 representative pilot frames. A suggested starting allocation is six development and four validation frames, adjusted for coverage and reported explicitly.
- [x] Extract by source timestamp using the verified task 1 conversion recipe. Record extraction timestamp, video ID, derivative dimensions, and condition tags. Never extract every frame as a default.
- [x] Produce labeled contact sheets and identify the documented shelf regions. Verify unknown-product examples against catalog identities, not shelf position alone.
- [ ] Annotate all visible display areas within explicit review ROIs using an existing local annotation tool if available; otherwise use image inspection and checked JSON coordinates for this small pilot. Save overlay renders for every annotation. This is reviewed seed data, not the full editor.
- [x] Record occupancy and observed/intended identity separately. Keep empty, mixed, and unreadable cases. Do not invent SKU labels from tray branding.
- [x] Test the schema invariants, out-of-bounds boxes, nonexistent SKUs, and assigning one video to both splits. Illustrative required assertions:

```python
def test_empty_area_cannot_claim_observed_sku():
    record = valid_area(occupancy="empty", identity_state="not_applicable",
                        observed_sku_id="sku-0001")
    with pytest.raises(ValueError):
        validate_annotation(record, catalog_ids={"sku-0001"})
```

The test file defines `valid_area` as a fixture factory matching the shared contract. Implement `validate_annotation(record: dict, catalog_ids: set[str]) -> None` in `schema.py`.

- [x] Run `uv run pytest tests/test_schema.py tests/test_pilot.py -q`; inspect overlays and correct labels before marking them reviewed. Exit artifact: pilot labels and split manifests ready for reproducible evaluation.

## Task 4: Run the local baseline

**Files:** `baseline.py`, `tests/test_baseline.py`, `configs/phase-0-baseline.json`.

**Interface:** `shelf-vision baseline --config configs/phase-0-baseline.json --data data/manifests --out reports/phase-0/baseline`.

- [x] Install and lock compatible model dependencies. Load Grounding DINO Tiny and DINOv2 ViT-S/14 sequentially, record checkpoint revisions, select MPS explicitly when supported, and log any CPU fallback and its cause.
- [ ] Test a small, recorded prompt set on development frames only, including display-oriented wording. Do not evaluate individual-package proposals as correct display-area detections without matching the annotation policy.
- [x] Embed normalized catalog crops. Rank each SKU by its maximum cosine similarity across approved references, then take top three distinct SKUs. Record this aggregation rule and gallery size.
- [x] Run recognition on verified occupied-area crops and on a paired visible-package-crop subset. Report the two conditions separately to expose the tray/package domain mismatch.
- [x] Run the full detection-to-retrieval path. Automated occupancy is absent: predictions are SKU candidates, not verified stock-presence claims.
- [x] Persist box proposals, ranked matches, timings, device, preprocessing, prompt, thresholds, and checkpoint provenance. Cache key includes source/crop identity, gallery hash, model revision, and inference configuration.
- [x] Unit-test embedding normalization, distinct-SKU ranking, empty-gallery errors, and cache invalidation for a changed prompt, crop, or gallery. Use fake vectors/adapters without downloading models in unit tests.
- [x] Run `uv run pytest tests/test_baseline.py -q`, then the real baseline CLI. Report cold start, warm inference, and whole-pipeline duration separately. Exit artifact: reproducible predictions and measured hardware behavior.

## Task 5: Evaluate and decide

**Files:** `evaluate.py`, `tests/test_evaluate.py`, `docs/phases/00-baseline-results.md`.

**Interface:** `shelf-vision evaluate --data data/manifests --predictions reports/phase-0/baseline --out reports/phase-0/evaluation`.

- [x] Match display-area detections one-to-one by descending detection confidence at IoU ≥ 0.5. Count duplicate matches as false positives. Assign each prediction to a review ROI by its box center. Ignore boxes centered outside reviewed ROIs or inside explicit ignore polygons. Clip evaluation boxes to their assigned ROI and apply the same clipping to ground truth; keep this policy fixed across runs.
- [x] Report detection precision/recall, recognition top-1/top-3 on verified known occupied regions, unknown false acceptance, known rejection, and end-to-end correctly localized and identified known regions. Include numerator and denominator for every metric.
- [x] Select any rejection threshold on development data only. Show validation results separately. Report both unknown false-accept rate and known-product acceptance at the same threshold so rejecting everything cannot look successful.
- [x] Exclude unresolved, empty, and mixed regions from single-SKU metrics and report their counts. If a required class is absent, emit `not measured` instead of zero or perfect accuracy.
- [x] Test duplicate detections, competing IoU matches, empty denominators, unresolved-versus-unknown labels, and a branded empty tray. Require zero observed-product credits for that tray.
- [x] Run `uv run pytest tests/test_evaluate.py -q`. Produce visual examples of misses, merges, wrong variants, and unknown errors, with a concise results document.
- [ ] Recommend keep/change for each component and set phase 1/2 numerical targets from evidence. If detector output is unusable, compare at most one practical alternative before choosing manual seed labeling. Do not hide a negative result behind UI work.

## Task 6: Check training/export feasibility and hand off

**Files:** `deployment_probe.py`, probe outputs under `reports/phase-0/`, updates to baseline results and `PROJECT.md`.

**Interface:** `shelf-vision deployment-probe --data data/manifests --out reports/phase-0/deployment`.

- [x] Export reviewed boxes to one-class YOLO labels, using upright dimensions and exact split membership. Check normalized coordinates and round-trip them to an overlay before training.
- [x] Run a deliberately short YOLO11n training probe on the reviewed development subset with a conservative batch size. Record memory, duration, device, configuration, and failures. The goal is executable training, not a performance claim.
- [x] Export the resulting checkpoint to Core ML on the Mac. Compare native and exported detections on the same few images with identical preprocessing/postprocessing; report matched boxes, score differences, and unmatched outputs. This is a measured parity report, not a fabricated universal tolerance.
- [x] Inspect a crowded and an oblique image. If export or inference fails, preserve the exact failure and name the required follow-up. Do not call deployment feasible based only on an export filename.
- [x] Run the focused unit suite once, rerun only affected checks after changes, and update the roadmap with results, catalog coverage, chosen baseline, remaining risks, and the next capture request.
- [ ] Hand off to phase 1 design. Keep physical iPhone performance and full training results explicitly outside this phase's claims.

## Execution notes

The completed annotation and metric steps use agent-reviewed candidates only. Human confirmation is still required. Only one detector prompt was measured; do not describe it as a prompt sweep. Raw Core ML parity is checked by `scripts/check_coreml_raw_parity.py`, because the initial short-run checkpoint produces no detections at 0.05 confidence. Numerical targets remain proposed in the results report.

## Completion record

Each task ends with its artifact paths, checks performed, results, and unresolved issues. Commit completed logical changes if Git has been initialized, staging named source/document files only. No milestone is complete merely because its commands ran: phase 0 exits with reviewed data, measured baselines, and a clear recommendation for the first product.
