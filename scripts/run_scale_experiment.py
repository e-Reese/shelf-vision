"""Fixed development comparison; existing snapshots and editor are read-only."""
import argparse
import html
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from shelf_vision.media import digest, save_json
from shelf_vision.scale_video import predict
from shelf_vision.training_metrics import detection_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, audit, out = args.snapshot.resolve(), args.audit.resolve(), args.out.resolve()
    locked = json.loads((source/'snapshot-lock.json').read_text())
    for name, sha in locked['files'].items():
        if name not in {'editor-snapshot.sqlite-wal','editor-snapshot.sqlite-shm'}:
            assert digest(source/name) == sha, name
    protocol = json.loads((source/'protocol.json').read_text())
    registration = protocol['candidate']
    assert digest(registration['checkpoint']) == registration['checkpoint_sha256']
    frames = json.loads((source/'manifests/frames.json').read_text())['frames']
    original = json.loads((source/'manifests/annotations.json').read_text())['annotations']
    revised = json.loads((audit/'candidate-annotations.json').read_text())['annotations']
    decisions = json.loads((audit/'audit.json').read_text())
    assert decisions['source_annotations_sha256'] == digest(source/'manifests/annotations.json')
    original_by_id = {a['region_id']:a for a in original}
    assert len({a['region_id'] for a in revised}) == len(revised)
    assert all(a == original_by_id[a['region_id']] for a in revised)
    frames = [{**f, 'split':'development'} for f in frames]
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    methods = ['whole640','whole1280','tiles960']
    config = {'methods':methods, 'confidence':.035, 'nms_iou':.5, 'tile_size':960,
              'overlap':.25, 'whole_pass_in_tiled':True, 'max_det':300, 'repeats':3,
              'gate':{'recall_gain':.10,'max_precision_loss':.05,'max_time_ratio':3},
              'selection_labels':'provisional_scope_filter', 'device':device,
              'checkpoint':registration['checkpoint'], 'checkpoint_sha256':registration['checkpoint_sha256'],
              'source_protocol_sha256':digest(source/'protocol.json'),
              'audit_sha256':digest(audit/'audit.json'), 'candidate_labels_sha256':digest(audit/'candidate-annotations.json'),
              'script_sha256':digest(Path(__file__)),
              'helper_sha256':{name:digest(Path(__file__).resolve().parents[1]/'src/shelf_vision'/name) for name in ['scale_video.py','training_metrics.py','evaluate.py','media.py']},
              'versions':{name:importlib.metadata.version(name) for name in ['torch','ultralytics','numpy','pillow']},
              'timing_scope':'Median of three warm complete dataset passes over preloaded PIL crops, including tile cropping, remapping, merging, and device synchronization; excludes disk decode, loading weights, and rendering.',
              'warning':'Examined visit now used for development. Tray audit is provisional; no deployment promotion or independent-test claim.'}
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'protocol.json', config)
    save_json(out/'audit.json', decisions)
    save_json(out/'candidate-annotations.json', {'annotations':revised})
    images = {}
    for f in frames:
        assert digest(f['path']) == f['crop_sha256']
        with Image.open(f['path']) as im:
            images[f['frame_id']] = im.convert('RGB')
    model = YOLO(registration['checkpoint'])
    predictions, timings = {}, {m:[] for m in methods}
    for method in methods:
        predict(model, images[frames[0]['frame_id']], method, device)
    for repeat in range(3):
        for method in methods[repeat:] + methods[:repeat]:
            if device == 'mps': torch.mps.synchronize()
            start = time.perf_counter()
            result = [{'frame_id':f['frame_id'], 'boxes':predict(model, images[f['frame_id']], method, device)} for f in frames]
            if device == 'mps': torch.mps.synchronize()
            timings[method].append(time.perf_counter()-start)
            if method not in predictions:
                predictions[method] = result
                save_json(out/(method+'-predictions.json'), {'frames':result})
            print(method, repeat, round(timings[method][-1],3), 'seconds', flush=True)
    groups = {'all':frames, 'close':[f for f in frames if '3523' in f['frame_id']],
              'wide':[f for f in frames if '3523' not in f['frame_id']],
              'manual':[f for f in frames if '3525' not in f['frame_id'] and '3526' not in f['frame_id']],
              'assisted':[f for f in frames if '3525' in f['frame_id'] or '3526' in f['frame_id']]}
    results = {}
    for label, truth in [('original',original),('provisional_scope_filter',revised)]:
        results[label] = {}
        for group, selected in groups.items():
            ids = {f['frame_id'] for f in selected}
            results[label][group] = {m:detection_metrics(selected, [a for a in truth if a['frame_id'] in ids],
                [p for p in predictions[m] if p['frame_id'] in ids], .035)['development'] for m in methods}
    medians = {m:statistics.median(timings[m]) for m in methods}
    reference = results['provisional_scope_filter']['all']['whole640']
    gates = {}
    for m in methods[1:]:
        row = results['provisional_scope_filter']['all'][m]
        checks = {'recall_gain':row['recall']-reference['recall'] >= .10,
                  'precision_loss':row['precision'] >= reference['precision']-.05,
                  'time_budget':medians[m] <= 3*medians['whole640']}
        gates[m] = {'checks':checks, 'passes_numeric_gate':all(checks.values()), 'time_ratio':medians[m]/medians['whole640']}
    eligible = [m for m in methods[1:] if gates[m]['passes_numeric_gate']]
    choice = max(eligible, key=lambda m:results['provisional_scope_filter']['all'][m]['f1']) if eligible else 'whole640'
    summary = {'metrics':results,'seconds_per_six_crops':timings,'median_seconds':medians,'gates':gates,
               'video_pilot_method':choice,'deployment_promoted':False,'labels_require_human_confirmation':True,
               'protocol_sha256':digest(out/'protocol.json')}
    save_json(out/'comparison.json', summary)
    panels = []
    for f in frames:
        fid = f['frame_id'];cells=[]
        for m, rows in [('provisional labels',[a for a in revised if a['frame_id']==fid])] + [(m,next(p['boxes'] for p in predictions[m] if p['frame_id']==fid)) for m in methods]:
            im = images[fid].copy(); draw=ImageDraw.Draw(im)
            for row in rows: draw.rectangle(row['bbox_xyxy'],outline='#40ff9a' if m=='provisional labels' else '#ff9944',width=4)
            im.thumbnail((700,900));name=fid+'-'+m.replace(' ','-')+'.jpg';im.save(out/name,quality=92)
            cells.append(f'<figure><figcaption>{m}: {len(rows)} boxes</figcaption><img src="{name}"></figure>')
        panels.append(f'<h2>{fid}</h2><section>{"".join(cells)}</section>')
    (out/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Scale experiment</title><style>body{font:15px system-ui;margin:24px}section{display:flex;gap:8px}figure{margin:0;flex:1;min-width:0}img{width:100%}pre{white-space:pre-wrap}</style><h1>Fixed scale experiment</h1><p>Development diagnostics. Filtered labels require human review. No model promoted.</p>'+''.join(panels)+'<pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>')
    print(json.dumps({'gates':gates,'choice':choice,'metrics':results['provisional_scope_filter']['all']},indent=2))


if __name__ == '__main__':
    main()
