# Experiment index

Shelf Vision investigates whether pretrained detection, embedding-based catalog retrieval, and a human review loop can support physical-tray detection on retail shelves.

The completed experiment ends with a tray-focused YOLO11n comparison: false positives fell from 45 to 7, but recall remained 13/134 on the selected validation crops. The predeclared improvement gate failed. The baseline editor configuration remains unchanged. Dense wide views, retrieval quality, and temporal tracking remain open problems.

## Reports

| Experiment | Evidence |
| --- | --- |
| Pretrained detection and catalog retrieval | [Baseline results](docs/phases/00-baseline-results.md) |
| Local annotation and review interface | [Editor verification](docs/phases/01-review-editor-results.md) |
| First reviewed-data YOLO training | [Training results](docs/phases/02-training-results.md) |
| Development-only threshold calibration | [Calibration results](docs/phases/02-calibration-results.md) |
| Locked second-visit evaluation | [Second-visit results](docs/phases/02-independent-visit-results.md) |
| Tray audit, image scale, and video pilot | [Scale and video results](docs/phases/02-scale-video-results.md) |
| Corrected-label YOLO training | [Final experiment results](docs/phases/02-tray-retraining-results.md) |

## Interpretation

The initial detection unit was a broad SKU display area. Later review clarified the target to one physical tray per box. Historical results retain the earlier labels and must not be compared directly with corrected-tray metrics. Retail packages and loose product groups without trays are excluded from the final target. Occupancy and product identity remain separate annotations.

The second visit was initially evaluated at a locked operating point. Once inspected and used to guide development, it was no longer an untouched test. The final experiment combines corrected labels from both visits with source-video-separated development and validation partitions. It does not establish performance at another store or on an unseen visit.

The final dataset contains 222 retained tray labels: 88 in 12 training crops and 134 in seven validation crops. Six training crops and one validation crop contain no trays. Review provenance distinguishes individual human decisions from inherited assistant judgments and proposal-assisted annotations.

## Reproduction boundaries

Code, seed configurations, tests, protocols, and measured summaries are public. Footage, reference images, editable databases, frozen datasets, generated reports, and checkpoints remain local. Paths under `data/`, `models/`, and `reports/` in the reports name local artifacts, not bundled downloads. See [usage](docs/USAGE.md) for setup and input requirements.

The detailed design and implementation notes document historical scope. Planned features in those notes are not claims of implemented functionality. Phone deployment, reliable video fusion, independent third-visit evaluation, and automatic inventory remain future work.
