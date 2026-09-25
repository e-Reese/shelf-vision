# Phase 0: Data and baseline design

Status: Historical baseline design. Engineering experiments run; human label review and phase acceptance remain open. See [results](00-baseline-results.md).

## Objective

Turn the supplied footage into a traceable pilot dataset and determine whether a local detector-plus-catalog-matcher can support the first review product. Produce evidence for model selection and phase 1 scope, without building the full application.

## Verified inventory

Source directory: `raw_footage/`. Original files remain unchanged.

| Files | Count | Observed metadata |
| --- | --- | --- |
| MOV videos | 4 | HEVC, stored at 3840×2160, approximately 59.97 fps, portrait rotation metadata, Dolby Vision/HDR metadata |
| HEIC photos | 121 | All report 2268×4032 pixels |
| Total | 125 | 398,433,260 bytes, approximately 398 MB |

| Video | Duration | Initial visual assessment from sampled frames |
| --- | --- | --- |
| IMG_3393.MOV | 7.535 s | Relatively close sweep across multiple shelf rows |
| IMG_3394.MOV | 9.637 s | Reverse/oblique sweep, with closer packaging and tray views |
| IMG_3395.MOV | 6.735 s | Wider angled view of multiple rows |
| IMG_3396.MOV | 7.602 s | Wider aisle/context view, with smaller product regions |

Total video duration: approximately 31.51 seconds. Video creation timestamps span less than one minute, consistent with one capture visit. Treat them as correlated development material, not independent visits.

Visual inspection covered 15 selected reference photos and approximately four sampled positions per video, not every photo or frame. Reference samples contain handheld front, side, and rear/barcode views against the store background. Product imagery often occupies only part of the photo. Some SKU variants look similar. Display trays, rails, reflections, and overlapping packages appear in the shelf footage.

The reviewer reports 40 documented products across two shelves. Other shelves are undocumented for unknown-product cases. The 121-photo count is consistent with roughly three views per product plus an extra, but this is not a verified grouping. Never assign product IDs by assuming fixed triplets.

## Approved detection unit: SKU display area

Detect one visually bounded display area or tray, not every individual package. A visible tray boundary defines the region where available; otherwise use the contiguous product group within a shelf row. Annotate the visible extent only and flag truncation or obscured boundaries. Do not infer invisible extents.

Use one detector class, `display_area`. Keep occupancy separate: `occupied`, `empty`, `mixed`, or `unclear`. Occupancy is human-reviewed metadata in phase 0, not a promised automated classifier.

Keep observed product identity separate from any intended SKU suggested by tray branding. An empty tray has no observed product SKU. A mixed area must not be forced into one SKU; mark it mixed and retain any individually verified identities as notes. Split adjoining groups only where a visible boundary supports the split.

Product identity states are `known`, `unknown`, `unresolved`, and `not_applicable`. A known identity requires a catalog ID and observable product evidence. Unknown means an observed product verified as outside the catalog. Empty and mixed areas have no single observed identity and use `not_applicable`; unclear occupancy uses `unresolved`.

Report display-area localization separately from product recognition. Exclude empty, mixed, and unresolved regions from single-SKU accuracy denominators and report their counts separately. Do not claim inventory or individual package counts.

## Data preparation design

1. Inventory originals with source-relative path, file size, checksum, media metadata, and capture group. Do not publish embedded location metadata.
2. Produce derived, correctly oriented RGB images. Verify HDR-to-SDR handling for video and color handling for HEIC before comparing embeddings. Inspection thumbnails are not training-ready exports.
3. Reconcile reference photos into 40 stable catalog IDs. Record visible name, variant, size, and barcode when readable; retain unresolved identity fields explicitly.
4. Crop the package from front/side reference photos to reduce store-background influence. Keep barcode/rear views as identity evidence; include them in retrieval only if an experiment supports their usefulness.
5. Choose about 10 pilot shelf frames across the four videos, prioritizing distinct views and conditions. Keep source filename and timestamp with every frame; retain full-resolution masters and generate smaller inference views separately.
6. Map the two documented shelf regions and verify which visible products are actually outside the catalog. Shelf location alone is not an unknown label: a catalog SKU could also appear elsewhere.
7. Human-review all target instances inside explicitly defined pilot regions. Mark catalog SKU, verified unknown, or unresolved identity separately. An unresolved identity is not confirmed unknown.

Derived media and manifests should live outside `raw_footage/`. Full directory structure and schemas belong in the implementation plan. Large media and model weights must be excluded from source control when Git is initialized.

## Development and evaluation split

Assign entire videos to development roles before extracting the pilot. Select roles after checking their coverage; do not randomly mix adjacent frames across splits. All current footage remains same-visit development evidence even when videos are separated.

Reserve a later visit as the final test. Use current development/validation material for prompt, crop, model, and rejection-threshold decisions. Keep unknown examples for both tuning and evaluation, and report limitations if their varieties overlap strongly.

## Baseline experiments

Run Grounding DINO Tiny for box proposals and DINOv2 ViT-S/14 for catalog retrieval. Confirm package versions, device selection, memory use, and fallback behavior on the Mac. No claimed M5 runtime until measured.

Evaluate the detector against reviewed boxes, including misses, duplicate detections, and boxes merging targets. Evaluate retrieval first on verified occupied-area crops, then on detector-produced crops. Compare whole-area crops with a manually verified visible-package crop subset: handheld package references and branded trays have a domain mismatch. If package crops work but area crops fail, record a need for package localization or another recognition approach rather than claiming the matcher is ready. Show top-1 and top-3 matches, known-product rejection, unknown false acceptance, and examples of similar-SKU confusion. Retrieval similarity remains a score, not a correctness probability.

Report end-to-end correct SKU detections and wall-clock processing time separately from isolated model inference time. Cache predictions using source identity, preprocessing settings, checkpoint identity, and relevant inference configuration.

Perform an early YOLO11n training/device smoke check on a small reviewed subset and a sample Core ML export/parity check on the Mac. These validate the engineering path, not model quality or iPhone performance. Full training belongs to phase 2, and physical phone acceptance belongs to phase 4.

## Deliverables and exit criteria

- Traceable media inventory, verified catalog mapping, and pilot frame/annotation manifests.
- Written annotation policy with the agreed detection unit and unknown/occlusion rules.
- Repeatable baseline command and saved predictions, configurations, timings, and error examples.
- Comparison report separating detection, catalog retrieval, and combined errors.
- Local training/export feasibility result, including any unsupported steps.
- Recommendation to keep the proposed stack or change a specific component, grounded in the pilot.
- Numerical phase 1/2 targets set from the pilot before iterative optimization, plus a next-visit capture list if needed.

If generic detection is inadequate, compare one practical alternative or create manual seed labels. Do not let model selection turn into an open-ended benchmark project. The next milestone remains a functioning review workflow.

## Remaining inputs

- Exact boundaries of the two documented shelves, established by visual mapping and confirmed where ambiguous.
- Verification of the 40 reference groups, particularly any similar variants or extra photos.
- A demo deadline if there is one; otherwise proceed by milestone completion.

The execution draft is [00-data-and-baseline-implementation.md](00-data-and-baseline-implementation.md). The phase 0 CLI and experiments have been implemented. The review application remains phase 1 work.
