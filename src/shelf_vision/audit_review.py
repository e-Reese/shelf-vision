"""Persistent, revision-checked audit decisions separate from frozen labels."""
import json
import sqlite3
from pathlib import Path
from typing import Literal
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from .media import digest
from .editor_store import Conflict, now


class AuditDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=0, strict=True)
    decision: Literal['tray','not_tray','needs_adjustment','pending']
    note: str = Field(default='', max_length=2000)


class AuditReview:
    def __init__(self, root, review="visit-002"):
        self.root = Path(root).resolve()
        if review not in {'visit-002', 'training'}:
            raise ValueError('Unknown audit review')
        self.review = review
        folder = 'training-tray-audit-001' if review == 'training' else 'tray-audit-002'
        path = self.root/'reports/phase-2'/folder/'audit.json'
        snapshot = (path.parent/'manifests' if review == 'training' else
                    self.root/'reports/phase-2/visit-002-evaluation/run-001/manifests')
        if not path.exists():
            raise ValueError('The tray audit is not available in this workspace.')
        self.sha = digest(path)
        audit = json.loads(path.read_text())
        if digest(snapshot/'annotations.json') != audit['source_annotations_sha256']:
            raise ValueError('The source annotations changed; restore the frozen snapshot.')
        annotations = {a['region_id']:a for a in json.loads((snapshot/'annotations.json').read_text())['annotations']}
        self.frames = {f['frame_id']:f for f in json.loads((snapshot/'frames.json').read_text())['frames']}
        self.items = []
        for row in audit['decisions']:
            a = annotations[row['region_id']]
            if a['frame_id'] != row['frame_id'] or row['frame_id'] not in self.frames:
                raise ValueError('Audit frame mismatch')
            self.items.append({**row, 'bbox_xyxy':a['bbox_xyxy'],
                               'flagged':row['status']=='boundary_or_tray_presence_uncertain',
                               'suggestion':'not_tray' if row['status'].startswith('exclude') else 'tray' if row['status']=='retain_candidate_tray' else 'pending'})
        if len({r['region_id'] for r in self.items}) != len(self.items):
            raise ValueError('Duplicate audit region')
        self.path = self.root/'data/editor'/('training-audit-review.sqlite' if review == 'training' else 'audit-review.sqlite')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS audit_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions(region_id TEXT PRIMARY KEY,decision TEXT NOT NULL,note TEXT NOT NULL,revision INTEGER NOT NULL,updated TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit_events(id INTEGER PRIMARY KEY,region_id TEXT NOT NULL,before_doc TEXT NOT NULL,after_doc TEXT NOT NULL,created TEXT NOT NULL);
            ''')
            db.execute('BEGIN IMMEDIATE')
            db.execute("INSERT OR IGNORE INTO audit_meta VALUES('source_sha',?)", (self.sha,))
            if db.execute("SELECT value FROM audit_meta WHERE key='source_sha'").fetchone()[0] != self.sha:
                raise ValueError('The source audit changed. Existing decisions were preserved; a new review version is required.')
            for r in self.items:
                decision = 'not_tray' if r['status']=='exclude_user_confirmed_nontray' else 'pending'
                db.execute('INSERT OR IGNORE INTO decisions VALUES(?,?,?,?,?)', (r['region_id'],decision,'',0,now()))

    def connect(self):
        # Caller closes explicitly through the wrapper; sqlite context alone only commits.
        from contextlib import contextmanager
        @contextmanager
        def connection():
            db=sqlite3.connect(self.path, timeout=30);db.row_factory=sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()
        return connection()

    def document(self, history=False):
        with self.connect() as db:
            db.execute('BEGIN')
            saved={r['region_id']:dict(r) for r in db.execute('SELECT * FROM decisions')}
            result={'review':self.review, 'source_audit_sha256':self.sha,
                    'frames':[{'frame_id':f['frame_id'],'width':f['width'],'height':f['height'],
                               'image_url':f"/api/audit/frames/{f['frame_id']}/image?review={self.review}"} for f in self.frames.values()],
                    'items':[{**r,**saved[r['region_id']]} for r in self.items]}
            if history:
                result['events']=[{'region_id':r['region_id'],'before':json.loads(r['before_doc']),
                                   'after':json.loads(r['after_doc']),'created':r['created']}
                                  for r in db.execute('SELECT * FROM audit_events ORDER BY id')]
        return result

    def save(self, region_id, request):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM decisions WHERE region_id=?',(region_id,)).fetchone()
            if row is None: raise KeyError(region_id)
            before=dict(row)
            if before['revision'] != request.revision:
                raise Conflict('This box was changed in another tab. Reload the review before saving again.')
            after={**before,'decision':request.decision,'note':request.note,
                   'revision':before['revision']+1,'updated':now()}
            db.execute('UPDATE decisions SET decision=?,note=?,revision=?,updated=? WHERE region_id=?',
                       (after['decision'],after['note'],after['revision'],after['updated'],region_id))
            db.execute('INSERT INTO audit_events(region_id,before_doc,after_doc,created) VALUES(?,?,?,?)',
                       (region_id,json.dumps(before),json.dumps(after),after['updated']))
        return after

    def image_path(self, frame_id):
        if frame_id not in self.frames: raise KeyError(frame_id)
        f=self.frames[frame_id];path=Path(f['path']).resolve()
        if not path.is_relative_to(self.root) or digest(path)!=f['crop_sha256']:
            raise ValueError('Audit image changed or is outside the workspace.')
        return path


def audit_router(root):
    router=APIRouter()

    @router.get('/audit')
    def page():
        return FileResponse(Path(__file__).with_name('audit.html'),headers={'Cache-Control':'no-store'})

    @router.get('/api/audit')
    def document(review: str = "visit-002"):
        return AuditReview(root, review).document()

    @router.get('/api/audit/export')
    def export(review: str = "visit-002"):
        return JSONResponse(AuditReview(root, review).document(history=True),headers={'Content-Disposition':'attachment; filename="tray-audit-decisions.json"'})

    @router.get('/api/audit/frames/{frame_id}/image')
    def image(frame_id: str, review: str = "visit-002"):
        return FileResponse(AuditReview(root, review).image_path(frame_id))

    @router.put('/api/audit/{region_id}')
    def save(region_id: str, body: AuditDecision, review: str = "visit-002"):
        return AuditReview(root, review).save(region_id,body)

    return router
