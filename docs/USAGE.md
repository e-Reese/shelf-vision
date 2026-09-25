# Local usage and reproduction

## Requirements

Use Python 3.12, uv, and Node.js 24. The captured-media preparation path uses macOS 15+, Swift/Xcode command-line tools, and ffprobe. Install the locked environments with `uv sync --locked` and `npm --prefix web ci`; build the UI with `npm --prefix web run build`.

The repository excludes original media, reference crops, databases, datasets, generated reports, and weights. Restoring those artifacts is necessary to reproduce the historical experiments. With new data, adapt the capture-specific selections and seed files before preparation. Do not reuse their coordinates against unrelated images.

## Prepare the original capture workflow

Place matching source media under `raw_footage/`, then run from the repository root:

```sh
uv run shelf-vision inventory --source raw_footage --out data/manifests/media.json
uv run shelf-vision prepare --manifest data/manifests/media.json --out data/derived/photos
uv run python scripts/build_catalog_seed.py
uv run python scripts/prepare_pilot.py
uv run python scripts/build_pilot_annotations.py
uv run shelf-vision catalog-validate --catalog data/manifests/catalog.json
uv run shelf-vision annotations-validate --data data/manifests
```

Seed labels are provisional and use the historical display-area policy. The final tray labels require the documented review and audit steps; preparation alone does not reconstruct the corrected dataset.

## Review images

```sh
uv run python -m shelf_vision.editor_api
```

Open http://127.0.0.1:8765 after the manifests and built frontend are present.

- B enables sticky box drawing; V selects and moves; H or Space + drag pans. The wheel zooms. Selected corners resize boxes, including while drawing mode is active.
- New boxes default to occupied and unknown. Set identity separately; an empty tray must not retain an observed product.
- Confirm frame saves the review, confirms coverage, and advances. Navigation saves pending edits without confirming coverage. Editing a confirmed frame requires reconfirmation.
- Proposals are suggestions. Explicitly accept useful boxes; rerunning a model preserves existing corrections.
- Imports accept JPEG, PNG, and HEIC up to 25 MB. Supply a capture group and split. Review the entire imported image or import a tighter crop.

SQLite state lives under `data/editor/`. Back up that directory with the server stopped. Browser drafts and revision conflicts have separate recovery behavior; do not assume browser shutdown saves an unsaved draft. Keep all frames from a source video in one split.

The tray audit pages `/audit` and `/audit?review=training` depend on the versioned artifacts named in `src/shelf_vision/audit_review.py`. Decisions save to separate databases. Needs adjustment records a follow-up, not a geometry edit.

## Baseline inference

```sh
HF_HOME="$PWD/models/huggingface" uv run shelf-vision baseline --config configs/phase-0-baseline.json --data data/manifests --out reports/phase-0/baseline
uv run shelf-vision evaluate --data data/manifests --predictions reports/phase-0/baseline/predictions.json --out reports/phase-0/evaluation
```

Pinned checkpoint revisions are recorded in configuration. First use may download weights. Retrieval similarities are not probabilities. Cached detector outputs retain original inference timings, not cache-read latency.

## Training experiments

The historical display-area pipeline freezes confirmed non-test frames and runs the original 0.25-threshold comparison:

```sh
uv run python -m shelf_vision.training_data --out data/datasets/reviewed-v2
HF_HOME="$PWD/models/huggingface" YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR="$PWD/models/ultralytics-config" uv run python -m shelf_vision.training --dataset data/datasets/reviewed-v2 --out reports/phase-2/experiment-002 --config configs/phase-2-training.json
```

Use fresh output paths. This historical runner is not the final corrected-tray experiment. The latter is implemented in `scripts/run_tray_retraining.py` and depends on the original frozen dataset, completed training audit, visit-2 audit version 3, and original checkpoint. Its `prepare` action writes the new dataset and protocol; `run` consumes one training budget. It refuses to overwrite an existing experiment. Input paths and hashes are version-specific; restoring artifacts to another checkout requires deliberate manifest migration, not bypassing integrity checks.

See the [final report](phases/02-tray-retraining-results.md) for the locked operating point, split, results, and limitations. Do not register the final checkpoint as an accepted replacement: it failed the specified gate.

## Verification

```sh
uv run pytest -q
npm --prefix web test
npm --prefix web run build
```

These checks do not require the private capture artifacts. They validate software behavior, not the reported model accuracy or real-device performance.
