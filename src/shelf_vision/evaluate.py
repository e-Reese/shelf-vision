"""Diagnostic metrics with explicit denominators and label provenance."""
import json
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
from .media import save_json
from .catalog import contact_sheet


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def match_boxes(predictions,truth,threshold=.5):
    used=set(); matches=[]
    for pi in sorted(range(len(predictions)),key=lambda i:-predictions[i]['confidence']):
        available=[(iou(predictions[pi]['bbox_xyxy'],g['bbox_xyxy']),gi) for gi,g in enumerate(truth) if gi not in used]
        if available:
            score,gi=max(available)
            if score>=threshold: matches.append((pi,gi));used.add(gi)
    return matches,len(predictions)-len(matches),len(truth)-len(matches)


def fraction(n,d): return {'numerator':n,'denominator':d,'rate':n/d if d else None}


def recognition_metrics(annotations,crops,threshold):
    by_id={c['region_id']:c for c in crops};known=unknown=top1=top3=accepted=fa=missing=excluded=0
    for a in annotations:
        if a['occupancy']!='occupied' or a['identity_state'] not in {'known','unknown'}:excluded+=1;continue
        if a['region_id'] not in by_id:missing+=1;continue
        matches=by_id[a['region_id']]['matches']; take=bool(matches) and matches[0]['score']>=threshold
        if a['identity_state']=='known':
            known+=1;accepted+=int(take)
            top1+=int(bool(matches) and matches[0]['sku_id']==a['observed_sku_id'])
            top3+=int(a['observed_sku_id'] in [m['sku_id'] for m in matches[:3]])
        else:unknown+=1;fa+=int(take)
    return {'top1':fraction(top1,known),'top3':fraction(top3,known),'known_acceptance':fraction(accepted,known),
            'unknown_false_accept':fraction(fa,unknown),'excluded':excluded,'missing_predictions':missing}


def clip(box,roi): return [max(box[0],roi[0]),max(box[1],roi[1]),min(box[2],roi[2]),min(box[3],roi[3])]

def inside(box,roi):
    x=(box[0]+box[2])/2;y=(box[1]+box[3])/2
    return roi[0]<=x<roi[2] and roi[1]<=y<roi[3]


def evaluate(data_dir,predictions_path,out):
    data_dir=Path(data_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    labels=json.loads((data_dir/'annotations.json').read_text())
    frames=json.loads((data_dir/'frames.json').read_text())['frames'];by_id={f['frame_id']:f for f in frames}
    predictions=json.loads(Path(predictions_path).read_text())
    crops=[c for c in predictions['crop_results'] if c['crop_kind']=='area']
    dev=[a for a in labels['annotations'] if by_id[a['frame_id']]['split']=='development']
    sweep=[]
    for threshold in [i/100 for i in range(0,101,5)]:
        metrics=recognition_metrics(dev,crops,threshold);sweep.append({'threshold':threshold,**metrics})
    eligible=[r for r in sweep if r['unknown_false_accept']['rate'] is not None and r['unknown_false_accept']['rate']<=.1]
    threshold=max(eligible,key=lambda r:(r['known_acceptance']['rate'] or 0,-r['threshold']))['threshold'] if eligible else None
    summary={'label_provenance':labels['review_status'],'human_review_status':labels['human_review_status'],
             'warning':'Provisional diagnostics on agent-reviewed annotations, NOT human-verified accuracy.',
             'rejection_threshold':threshold,'threshold_selection':'development only; maximize known acceptance with unknown false acceptance <= 0.1',
             'development_threshold_sweep':sweep,'splits':{}}
    for split in ['development','validation']:
        truth=[a for a in labels['annotations'] if by_id[a['frame_id']]['split']==split]
        tp=fp=fn=correct=0
        for f in [f for f in frames if f['split']==split]:
            prediction=next(p for p in predictions['frames'] if p['frame_id']==f['frame_id'])
            for roi in [r for r in labels['rois'] if r['frame_id']==f['frame_id']]:
                gt=[{**a,'bbox_xyxy':clip(a['bbox_xyxy'],roi['bbox_xyxy'])} for a in truth if a['review_roi_id']==roi['roi_id']]
                pp=[{**p,'bbox_xyxy':clip(p['bbox_xyxy'],roi['bbox_xyxy'])} for p in prediction['boxes'] if inside(p['bbox_xyxy'],roi['bbox_xyxy'])]
                matches,extra,miss=match_boxes(pp,gt);tp+=len(matches);fp+=extra;fn+=miss
                for pi,gi in matches:
                    a=gt[gi];m=pp[pi].get('matches',[])
                    if a['identity_state']=='known' and m and m[0]['sku_id']==a['observed_sku_id']:correct+=1
        summary['splits'][split]={'detection_precision':fraction(tp,tp+fp),'detection_recall':fraction(tp,tp+fn),
            'end_to_end_top1_recall_before_rejection':fraction(correct,sum(a['identity_state']=='known' for a in truth)),
            'retrieval_on_area_crops':recognition_metrics(truth,crops,threshold if threshold is not None else 1.1),
            'occupancy_counts':dict(Counter(a['occupancy'] for a in truth))}
    sheets=[]
    for f,p in zip(frames,predictions['frames']):
        im=Image.open(f['path']).convert('RGB');im.thumbnail((810,1440));draw=ImageDraw.Draw(im);sx=im.width/f['width'];sy=im.height/f['height']
        for b in p['boxes']:
            box=[v*([sx,sy][i%2]) for i,v in enumerate(b['bbox_xyxy'])];draw.rectangle(box,outline='red',width=2)
            match=b.get('matches',[{}])[0];draw.text((box[0],box[1]),f"{match.get('sku_id','?')} {match.get('score',0):.2f}",fill='white',stroke_width=1,stroke_fill='black')
        path=out/(f['frame_id']+'.jpg');im.save(path,quality=95);sheets.append({'path':str(path),'label':f['frame_id']+' raw model proposals; not verified occupancy'})
    contact_sheet(sheets,out/'predictions.html');save_json(out/'metrics.json',summary)
    return summary
