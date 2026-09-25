"""Transactional local review state. Source proposals are never modified by saves."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid
from .schema import validate_annotation


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, allow_nan=False)


class Conflict(Exception):
    pass


class EditorStore:
    def __init__(self, path, catalog):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        with self.connection() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS frames(id TEXT PRIMARY KEY,meta TEXT NOT NULL,annotations TEXT NOT NULL,proposals TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 0,status TEXT NOT NULL DEFAULT 'unreviewed',updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,frame_id TEXT NOT NULL,revision INTEGER NOT NULL,action TEXT NOT NULL,created TEXT NOT NULL,before_doc TEXT NOT NULL,after_doc TEXT NOT NULL,changed_ids TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY,frame_id TEXT NOT NULL,created TEXT NOT NULL,result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,frame_id TEXT NOT NULL,status TEXT NOT NULL,error TEXT,created TEXT NOT NULL,updated TEXT NOT NULL);
            """)
            if "model" not in {
                r["name"] for r in db.execute("PRAGMA table_info(jobs)")
            }:
                db.execute(
                    "ALTER TABLE jobs ADD COLUMN model TEXT NOT NULL DEFAULT 'baseline'"
                )

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_frame(self, meta, annotations, proposals, enforce_group=False):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if enforce_group:
                for row in db.execute("SELECT meta FROM frames"):
                    existing = json.loads(row["meta"])
                    if (
                        existing["capture_group"] == meta["capture_group"]
                        and existing["split"] != meta["split"]
                    ):
                        raise ValueError(
                            "This capture group already has a different split. Use its existing split or a genuinely different capture group."
                        )
            db.execute(
                "INSERT OR IGNORE INTO frames(id,meta,annotations,proposals,updated) VALUES(?,?,?,?,?)",
                (
                    meta["frame_id"],
                    encode(meta),
                    encode(annotations),
                    encode(proposals),
                    now(),
                ),
            )
        return self.get_frame(meta["frame_id"])

    def list_frames(self):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM frames ORDER BY rowid").fetchall()
        return [
            {
                **json.loads(r["meta"]),
                "path": None,
                "revision": r["revision"],
                "status": r["status"],
                "area_count": len(json.loads(r["annotations"])),
                "updated": r["updated"],
            }
            for r in rows
        ]

    def get_frame(self, frame_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone()
            run = db.execute(
                "SELECT * FROM runs WHERE frame_id=? ORDER BY id DESC LIMIT 1",
                (frame_id,),
            ).fetchone()
        if row is None:
            raise KeyError(frame_id)
        original = json.loads(row["proposals"])
        result = json.loads(run["result"]) if run else None
        return {
            "meta": json.loads(row["meta"]),
            "annotations": json.loads(row["annotations"]),
            "original_proposals": original,
            "proposals": result["boxes"] if result else original,
            "prediction_provenance": result.get("provenance")
            if result
            else {"source": "phase-0 baseline"},
            "revision": row["revision"],
            "status": row["status"],
            "updated": row["updated"],
        }

    def save(self, frame_id, annotations, revision, reviewed=False):
        if not isinstance(annotations, list) or len(annotations) > 2000:
            raise ValueError("Invalid annotation list")
        if not isinstance(revision, int) or isinstance(revision, bool):
            raise ValueError("Invalid revision")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone()
            if row is None:
                raise KeyError(frame_id)
            if row["revision"] != revision:
                raise Conflict(
                    "A newer version exists. Your draft was kept. Reload the server version before saving again."
                )
            meta = json.loads(row["meta"])
            rois = {r["roi_id"]: r for r in meta["rois"]}
            ids = set()
            clean = []
            catalog_ids = {p["sku_id"] for p in self.catalog}
            for a in annotations:
                if not isinstance(a, dict):
                    raise ValueError("Invalid annotation")
                rid = a.get("region_id")
                if not isinstance(rid, str) or not rid or len(rid) > 200 or rid in ids:
                    raise ValueError("Area IDs must be present and unique")
                ids.add(rid)
                if a.get("frame_id") != frame_id or a.get("review_roi_id") not in rois:
                    raise ValueError("Area belongs to another frame or review region")
                validate_annotation(a, catalog_ids, meta["width"], meta["height"])
                x1, y1, x2, y2 = a["bbox_xyxy"]
                roi = rois[a["review_roi_id"]]["bbox_xyxy"]
                if not (roi[0] <= x1 < x2 <= roi[2] and roi[1] <= y1 < y2 <= roi[3]):
                    raise ValueError("Area must remain inside its review region")
                fields = [
                    "region_id",
                    "frame_id",
                    "review_roi_id",
                    "bbox_xyxy",
                    "class_name",
                    "occupancy",
                    "identity_state",
                    "observed_sku_id",
                    "intended_sku_id",
                    "truncated",
                ]
                clean.append(
                    {
                        **{k: a.get(k) for k in fields},
                        "review_status": "human_reviewed" if reviewed else "draft",
                        "human_review_status": "reviewed" if reviewed else "pending",
                    }
                )
            before = json.loads(row["annotations"])
            old = {a["region_id"]: a for a in before}
            new = {a["region_id"]: a for a in clean}
            changed = [
                i for i in sorted(set(old) | set(new)) if old.get(i) != new.get(i)
            ]
            timestamp = now()
            status = "reviewed" if reviewed else "in_progress"
            db.execute(
                "UPDATE frames SET annotations=?,revision=?,status=?,updated=? WHERE id=?",
                (encode(clean), revision + 1, status, timestamp, frame_id),
            )
            db.execute(
                "INSERT INTO events(frame_id,revision,action,created,before_doc,after_doc,changed_ids) VALUES(?,?,?,?,?,?,?)",
                (
                    frame_id,
                    revision + 1,
                    "finish_review" if reviewed else "save",
                    timestamp,
                    encode(before),
                    encode(clean),
                    encode(changed),
                ),
            )
        return self.get_frame(frame_id)

    def events(self, frame_id):
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM events WHERE frame_id=? ORDER BY id", (frame_id,)
            ).fetchall()
        return [
            {
                **dict(r),
                "before_doc": json.loads(r["before_doc"]),
                "after_doc": json.loads(r["after_doc"]),
                "changed_ids": json.loads(r["changed_ids"]),
            }
            for r in rows
        ]

    def add_prediction_run(self, frame_id, result):
        self.get_frame(frame_id)
        with self.connection() as db:
            db.execute(
                "INSERT INTO runs(frame_id,created,result) VALUES(?,?,?)",
                (frame_id, now(), encode(result)),
            )

    def runs(self, frame_id):
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM runs WHERE frame_id=? ORDER BY id", (frame_id,)
            ).fetchall()
        return [{**dict(r), "result": json.loads(r["result"])} for r in rows]

    def reviewed_snapshot(self):
        """Freeze documents, proposals, and audit history in one read transaction."""
        records = []
        with self.connection() as db:
            db.execute("BEGIN")
            for row in db.execute(
                "SELECT * FROM frames WHERE status='reviewed' ORDER BY rowid"
            ).fetchall():
                meta = json.loads(row["meta"])
                if meta["split"] == "test":
                    continue
                events = []
                for event in db.execute(
                    "SELECT * FROM events WHERE frame_id=? ORDER BY id", (row["id"],)
                ):
                    event = dict(event)
                    for key in ["before_doc", "after_doc", "changed_ids"]:
                        event[key] = json.loads(event[key])
                    events.append(event)
                runs = [
                    {**dict(r), "result": json.loads(r["result"])}
                    for r in db.execute(
                        "SELECT * FROM runs WHERE frame_id=? ORDER BY id", (row["id"],)
                    )
                ]
                records.append(
                    {
                        "meta": meta,
                        "revision": row["revision"],
                        "annotations": json.loads(row["annotations"]),
                        "original_proposals": json.loads(row["proposals"]),
                        "events": events,
                        "runs": runs,
                    }
                )
        return records

    def create_job(self, frame_id, model="baseline"):
        self.get_frame(frame_id)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM jobs WHERE frame_id=? AND status IN ('queued','running')",
                (frame_id,),
            ).fetchone()
            if existing:
                if existing["model"] != model:
                    raise Conflict(
                        "Another model is already running for this frame. Wait for it to finish."
                    )
                return dict(existing), False
            jid = str(uuid.uuid4())
            ts = now()
            db.execute(
                "INSERT INTO jobs(id,frame_id,status,error,created,updated,model) VALUES(?,?,?,?,?,?,?)",
                (jid, frame_id, "queued", None, ts, ts, model),
            )
        return self.job(jid), True

    def job(self, jid):
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if row is None:
            raise KeyError(jid)
        return dict(row)

    def update_job(self, jid, status, error=None):
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status=?,error=?,updated=? WHERE id=?",
                (status, error, now(), jid),
            )

    def recover_jobs(self):
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status='failed',error='Server restarted. Please run suggestions again.',updated=? WHERE status IN ('queued','running')",
                (now(),),
            )


def seed_pilot(store, root):
    root = Path(root)
    base = root / "data/manifests"
    if not (base / "frames.json").exists():
        return
    frames = json.loads((base / "frames.json").read_text())["frames"]
    labels = (
        json.loads((base / "annotations.json").read_text())
        if (base / "annotations.json").exists()
        else {"annotations": [], "rois": []}
    )
    path = root / "reports/phase-0/baseline/predictions.json"
    predictions = json.loads(path.read_text())["frames"] if path.exists() else []
    for frame in frames:
        fid = frame["frame_id"]
        rois = [r for r in labels["rois"] if r["frame_id"] == fid]
        meta = {
            **frame,
            "name": f"{frame['video_path']} · {frame['requested_seconds']:g}s",
            "source_id": frame["video_id"],
            "rois": rois
            or [
                {
                    "roi_id": fid + "-full",
                    "bbox_xyxy": [0, 0, frame["width"], frame["height"]],
                }
            ],
        }
        annotations = [a for a in labels["annotations"] if a["frame_id"] == fid]
        proposals = next((r["boxes"] for r in predictions if r["frame_id"] == fid), [])
        store.add_frame(meta, annotations, proposals)
