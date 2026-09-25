"""One declared tray-label experiment, with immutable inputs and paired evaluation."""
import argparse
import html
import importlib.metadata
import json
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from shelf_vision.audit_review import AuditReview
from shelf_vision.media import digest, save_json
from shelf_vision.training_data import validate_dataset
from shelf_vision.training_metrics import detection_metrics


def verify(path):
    for name, sha in json.loads((path/'lock.json').read_text())['files'].items():
        if digest(path/name) != sha:
            raise ValueError('Frozen input changed: '+name)


def prepare(root, dataset, out):
    old=root/'data/datasets/reviewed-v1'
    validate_dataset(old)
    review=AuditReview(root,'training').document(history=True)
    if any(x['decision'] not in {'tray','not_tray'} for x in review['items'] if x['flagged']) or any(x['decision']=='needs_adjustment' for x in review['items']):
        raise ValueError('Resolve all flagged cases and boundary corrections before training')
    old_frames=json.loads((old/'manifests/frames.json').read_text())['frames']
    old_labels=json.loads((old/'manifests/annotations.json').read_text())['annotations']
    keep={x['region_id'] for x in review['items'] if x['decision']=='tray' or (x['decision']=='pending' and x['suggestion']=='tray')}
    original=[a for a in old_labels if a['region_id'] in keep]
    visit=root/'reports/phase-2/tray-audit-003'
    result=json.loads((visit/'rescored.json').read_text())
    assert digest(visit/'candidate-annotations.json')==result['annotations_sha256']
    assert digest(visit/'audit.json')==result['audit_sha256']
    snap=root/'reports/phase-2/visit-002-evaluation/run-001/manifests'
    frames2=json.loads((snap/'frames.json').read_text())['frames']
    for f in frames2:
        assert digest(f['path'])==f['crop_sha256']
        f['split']='development' if f['video_path'] in {'IMG_3523.MOV','IMG_3524.MOV'} else 'validation'
    labels2=json.loads((visit/'candidate-annotations.json').read_text())['annotations']
    frames=old_frames+frames2;labels=original+labels2
    assert len({f['frame_id'] for f in frames})==len(frames)
    assert len({a['region_id'] for a in labels})==len(labels)
    source_splits={}
    for f in frames:
        source=f['video_id']
        assert source_splits.setdefault(source,f['split'])==f['split'], 'Source video leakage'
    dataset.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    for split in ('train','val'):
        (dataset/'images'/split).mkdir(parents=True);(dataset/'labels'/split).mkdir(parents=True)
    for f in frames:
        split='train' if f['split']=='development' else 'val';src=Path(f['path']);dst=dataset/'images'/split/(f['frame_id']+'.png')
        shutil.copy2(src,dst);f['source_path']=str(src);f['source_sha256']=digest(src);f['path']=str(dst)
        with Image.open(dst) as image:assert image.size==(f['width'],f['height'])
        rows=[]
        for a in labels:
            if a['frame_id']!=f['frame_id']:continue
            x1,y1,x2,y2=a['bbox_xyxy'];w,h=f['width'],f['height']
            assert 0<=x1<x2<=w and 0<=y1<y2<=h
            rows.append(f'0 {(x1+x2)/2/w:.9f} {(y1+y2)/2/h:.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}')
        (dataset/'labels'/split/(f['frame_id']+'.txt')).write_text('\n'.join(rows)+ ('\n' if rows else ''))
    (dataset/'manifests').mkdir()
    save_json(dataset/'manifests/frames.json',{'frames':frames});save_json(dataset/'manifests/annotations.json',{'annotations':labels})
    save_json(dataset/'training-review.json',review)
    shutil.copy2(visit/'audit.json',dataset/'visit2-audit.json')
    shutil.copy2(visit/'review-decisions.json',dataset/'visit2-review.json')
    (dataset/'data.yaml').write_text(f'path: {json.dumps(str(dataset))}\ntrain: images/train\nval: images/val\nnames:\n  0: tray\n')
    counts={s:{'images':sum(f['split']==s for f in frames),'trays':sum(a['frame_id'] in {f['frame_id'] for f in frames if f['split']==s} for a in labels),'negative_images':sum(f['split']==s and not any(a['frame_id']==f['frame_id'] for a in labels) for f in frames)} for s in ('development','validation')}
    save_json(dataset/'lock.json',{'counts':counts,'files':{str(p.relative_to(dataset)):digest(p) for p in dataset.rglob('*') if p.is_file()}})
    config=json.loads((root/'configs/phase-2-training.json').read_text())
    checkpoints={'initial':root/'yolo11n.pt','old':root/'reports/phase-2/experiment-001/train/weights/best.pt'}
    protocol={'dataset':str(dataset),'dataset_lock_sha256':digest(dataset/'lock.json'),'counts':counts,'train':config['train'],'evaluation':{'confidence':.035,'nms_iou':.5,'matching_iou':.5,'imgsz':640,'rect':False,'max_det':300},'gate':{'recall_gain':.10,'max_precision_loss':.05,'max_time_ratio':3},'checkpoints':{k:{'path':str(v),'sha256':digest(v)} for k,v in checkpoints.items()},'script_sha256':digest(Path(__file__)),'helper_sha256':{name:digest(root/'src/shelf_vision'/name) for name in ['training_metrics.py','evaluate.py','audit_review.py']},'versions':{n:importlib.metadata.version(n) for n in ['torch','ultralytics','numpy','pillow']},'run_budget':1,'timing':'Three alternating warm validation passes with preloaded PIL images and device synchronization. Disk decode and model load excluded.','warning':'Previously inspected development validation, not independent new-visit testing. Mixed assistant and human label review; visit2 validation labels were baseline-assisted. No automatic deployment promotion.'}
    save_json(out/'protocol.json',protocol)
    verify(dataset)
    print(json.dumps(counts),flush=True)


def run(root,dataset,out):
    import torch
    from ultralytics import YOLO
    protocol=json.loads((out/'protocol.json').read_text())
    assert digest(dataset/'lock.json')==protocol['dataset_lock_sha256']
    assert digest(Path(__file__))==protocol['script_sha256']
    verify(dataset)
    for c in protocol['checkpoints'].values():assert digest(c['path'])==c['sha256']
    if (out/'experiment.json').exists():raise ValueError('One-run budget already consumed')
    record={'status':'running','protocol_sha256':digest(out/'protocol.json'),'device':'mps' if torch.backends.mps.is_available() else 'cpu'}
    save_json(out/'experiment.json',record);device=record['device']
    try:
        model=YOLO(protocol['checkpoints']['initial']['path']);start=time.perf_counter()
        model.train(data=str(dataset/'data.yaml'),device=device,project=str(out),name='train',exist_ok=False,plots=False,**protocol['train'])
        record['training_seconds']=time.perf_counter()-start
        checkpoint=out/'train/weights/best.pt';record['checkpoint_sha256']=digest(checkpoint)
        frames=json.loads((dataset/'manifests/frames.json').read_text())['frames'];labels=json.loads((dataset/'manifests/annotations.json').read_text())['annotations']
        val=[f for f in frames if f['split']=='validation'];images={}
        for f in val:
            with Image.open(f['path']) as im:images[f['frame_id']]=im.convert('RGB')
        models={'old':YOLO(protocol['checkpoints']['old']['path']),'new':YOLO(str(checkpoint))}
        def predict(model,f):
            result=model.predict(images[f['frame_id']],imgsz=640,conf=.035,iou=.5,rect=False,max_det=300,device=device,verbose=False)[0]
            return {'frame_id':f['frame_id'],'boxes':[{'bbox_xyxy':r[:4],'confidence':r[4]} for r in result.boxes.data.cpu().tolist()]}
        for model in models.values():predict(model,val[0])
        predictions={};times={'old':[],'new':[]}
        for repeat in range(3):
            for name in (['old','new'] if repeat%2==0 else ['new','old']):
                if device=='mps':torch.mps.synchronize()
                start=time.perf_counter();rows=[predict(models[name],f) for f in val]
                if device=='mps':torch.mps.synchronize()
                times[name].append(time.perf_counter()-start)
                if name not in predictions:
                    predictions[name]=rows;save_json(out/(name+'-predictions.json'),{'frames':rows})
        groups={'all':val,'visit1_close_oblique':[f for f in val if f['capture_group']=='visit-001'],'visit2_wide_assisted':[f for f in val if f['capture_group']=='visit-002'],'negative_crops':[f for f in val if not any(a['frame_id']==f['frame_id'] for a in labels)]}
        metrics={}
        for group,selected in groups.items():
            ids={f['frame_id'] for f in selected}
            if selected:metrics[group]={name:detection_metrics(selected,[a for a in labels if a['frame_id'] in ids],[p for p in predictions[name] if p['frame_id'] in ids],.035)['validation'] for name in models}
        old,new=metrics['all']['old'],metrics['all']['new'];ratio=statistics.median(times['new'])/statistics.median(times['old'])
        checks={'recall_gain':new['recall']-old['recall']>=.10,'precision_loss':new['precision']>=old['precision']-.05,'time_budget':ratio<=3}
        comparison={'metrics':metrics,'timings':times,'time_ratio':ratio,'gate':{'checks':checks,'passes':all(checks.values())},'deployment_promoted':False,'warning':protocol['warning'],'protocol_sha256':digest(out/'protocol.json'),'prediction_sha256':{n:digest(out/(n+'-predictions.json')) for n in models}}
        save_json(out/'comparison.json',comparison)
        panels=[]
        for f in val:
            cells=[]
            for name in models:
                im=images[f['frame_id']].copy();d=ImageDraw.Draw(im)
                for a in labels:
                    if a['frame_id']==f['frame_id']:d.rectangle(a['bbox_xyxy'],outline='#55ee88',width=5)
                for b in next(p['boxes'] for p in predictions[name] if p['frame_id']==f['frame_id']):d.rectangle(b['bbox_xyxy'],outline='#ff8844',width=3)
                im.thumbnail((850,850));file=f['frame_id']+'-'+name+'.jpg';im.save(out/file,quality=92);cells.append(f'<figure><figcaption>{name}</figcaption><img src="{file}"></figure>')
            panels.append('<h2>'+f['frame_id']+'</h2><section>'+''.join(cells)+'</section>')
        (out/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Tray retraining comparison</title><style>body{font:16px system-ui;margin:30px}section{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%}pre{white-space:pre-wrap}</style><h1>Tray retraining comparison</h1><p>Development validation. Green: tray labels. Orange: predictions. No deployment promotion.</p><pre>'+html.escape(json.dumps(comparison,indent=2))+'</pre>'+''.join(panels))
        verify(dataset);record['status']='complete';record['gate']=comparison['gate'];save_json(out/'experiment.json',record)
        print(json.dumps({'metrics':metrics['all'],'gate':comparison['gate'],'time_ratio':ratio}),flush=True)
    except Exception as exc:
        record.update(status='failed',error=str(exc));save_json(out/'experiment.json',record);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];dataset=root/'data/datasets/trays-v2';out=root/'reports/phase-2/tray-experiment-002'
    (prepare if args.action=='prepare' else run)(root,dataset,out)
