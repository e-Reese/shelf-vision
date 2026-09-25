"""Small engineering smoke test, explicitly not an accuracy-training experiment."""
import json
import time
from pathlib import Path
from PIL import Image,ImageDraw
from .media import save_json
from .evaluate import clip,match_boxes
from .schema import validate_splits


def fresh_dataset_directory(out):
    import tempfile
    Path(out).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='dataset-', dir=out))


def parity_predict(model, path):
    return model(str(path), imgsz=640, device='cpu', conf=.05, rect=False, verbose=False)[0]


def yolo_box(box,roi):
    x1,y1,x2,y2=clip(box,roi);w=roi[2]-roi[0];h=roi[3]-roi[1]
    if w<=0 or h<=0 or x2<=x1 or y2<=y1: raise ValueError('Box does not overlap ROI')
    return [((x1+x2)/2-roi[0])/w,((y1+y2)/2-roi[1])/h,(x2-x1)/w,(y2-y1)/h]


def deployment_probe(data_dir,out):
    import torch
    from ultralytics import YOLO
    data_dir=Path(data_dir);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    frames=json.loads((data_dir/'frames.json').read_text())['frames'];validate_splits(frames);by_id={f['frame_id']:f for f in frames}
    labels=json.loads((data_dir/'annotations.json').read_text())
    dataset=fresh_dataset_directory(out)
    for roi in labels['rois']:
        f=by_id[roi['frame_id']];split='train' if f['split']=='development' else 'val'
        if f['split']=='test': raise ValueError('Final test data must not enter smoke training')
        for kind in ['images','labels']:(dataset/kind/split).mkdir(parents=True,exist_ok=True)
        image=Image.open(f['path']).convert('RGB').crop(roi['bbox_xyxy'])
        image.save(dataset/'images'/split/(roi['roi_id']+'.jpg'),quality=95)
        lines=[];overlay=image.copy();draw=ImageDraw.Draw(overlay)
        for a in labels['annotations']:
            if a['review_roi_id']!=roi['roi_id']:continue
            coords=yolo_box(a['bbox_xyxy'],roi['bbox_xyxy'])
            lines.append('0 '+' '.join(f'{v:.8f}' for v in coords))
            x,y,w,h=coords;draw.rectangle([(x-w/2)*image.width,(y-h/2)*image.height,(x+w/2)*image.width,(y+h/2)*image.height],outline='red',width=3)
        (dataset/'labels'/split/(roi['roi_id']+'.txt')).write_text('\n'.join(lines)+'\n')
        overlay.thumbnail((1000,1000));overlay.save(out/(roi['roi_id']+'-roundtrip.jpg'))
    yaml=dataset/'data.yaml';yaml.write_text(f'path: {dataset}\ntrain: images/train\nval: images/val\nnames:\n  0: display_area\n')
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    record={'purpose':'engineering smoke test only','label_provenance':'agent_reviewed; human review pending','device':device,'epochs':3,'imgsz':640,'batch':2,'seed':0,'dataset':str(dataset)}
    save_json(out/'result.json',record)
    model=YOLO('yolo11n.pt');t=time.perf_counter()
    model.train(data=str(yaml),epochs=3,imgsz=640,batch=2,device=device,workers=0,project=str(out),name='train',exist_ok=True,plots=False,seed=0,amp=False,cache=False,mosaic=0,fliplr=0)
    record['training_seconds']=time.perf_counter()-t
    best=out/'train/weights/best.pt';record['checkpoint']=str(best)
    save_json(out/'result.json',record)
    trained=YOLO(str(best))
    try:
        t=time.perf_counter();package=trained.export(format='coreml',imgsz=640,nms=True,half=False)
        record['export_seconds']=time.perf_counter()-t;record['coreml_package']=str(package)
        exported=YOLO(str(package));parity=[]
        for path in sorted((dataset/'images/val').glob('*.jpg'))[:3]:
            native=parity_predict(trained,path)
            converted=parity_predict(exported,path)
            def boxes(result):return [{'bbox_xyxy':b[:4],'confidence':b[4]} for b in result.boxes.data.cpu().tolist()]
            nb,cb=boxes(native),boxes(converted);matches,fp,fn=match_boxes(cb,nb)
            parity.append({'image':path.name,'native_count':len(nb),'coreml_count':len(cb),'matched_iou_0_5':len(matches),'coreml_unmatched':fp,'native_unmatched':fn,
                           'mean_abs_score_difference':sum(abs(cb[i]['confidence']-nb[j]['confidence']) for i,j in matches)/len(matches) if matches else None})
        record['parity']=parity;record['export_status']='export_and_inference_executed'
        record['parity_status']='inconclusive_no_detections' if not any(p['native_count'] or p['coreml_count'] for p in parity) else 'measured_requires_review'
    except Exception as exc:
        record['export_status']='failed';record['export_error']=f'{type(exc).__name__}: {exc}'
    save_json(out/'result.json',record);return record
