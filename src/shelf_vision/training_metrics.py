"""Fixed operating-point metrics shared by compared detectors."""

from .evaluate import match_boxes


def detection_metrics(frames, annotations, predictions, confidence=0.25):
    by_id = {p["frame_id"]: p for p in predictions}
    expected = {f["frame_id"] for f in frames}
    if len(by_id) != len(predictions) or set(by_id) != expected:
        raise ValueError("Missing, duplicate or unexpected prediction frame")
    if any(a["frame_id"] not in expected for a in annotations):
        raise ValueError("Unknown annotation frame")
    result = {}
    for split in sorted({f["split"] for f in frames}):
        rows = []
        totals = {"known": [0, 0], "unknown": [0, 0]}
        for f in [f for f in frames if f["split"] == split]:
            gt = [a for a in annotations if a["frame_id"] == f["frame_id"]]
            pred = [
                b
                for b in by_id[f["frame_id"]]["boxes"]
                if b["confidence"] >= confidence
            ]
            pairs, fp, fn = match_boxes(pred, gt, 0.5)
            matched = {gi for _, gi in pairs}
            for i, a in enumerate(gt):
                kind = a["identity_state"]
                if kind in totals:
                    totals[kind][0] += int(i in matched)
                    totals[kind][1] += 1
            rows.append(
                {
                    "frame_id": f["frame_id"],
                    "tp": len(pairs),
                    "fp": fp,
                    "fn": fn,
                    "truth_count": len(gt),
                    "prediction_count": len(pred),
                }
            )
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        result[split] = {
            "images": len(rows),
            "objects": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "per_image": rows,
        }
        for kind, (n, d) in totals.items():
            result[split][kind + "_recall"] = {
                "matched": n,
                "total": d,
                "rate": n / d if d else None,
            }
    return result


def promotion_gate(baseline, candidate):
    checks = {
        "f1_gain_at_least_0_10": candidate["f1"] - baseline["f1"] >= 0.10 - 1e-12,
        "recall_not_lower": candidate["recall"] >= baseline["recall"],
        "precision_at_least_0_50": candidate["precision"] >= 0.50,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "f1_gain": candidate["f1"] - baseline["f1"],
    }
