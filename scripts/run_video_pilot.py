"""Short, reproducible tracking feasibility demo, without accuracy claims."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from shelf_vision.media import digest, extract_frame, save_json
from shelf_vision.scale_video import Tracker, camera_motion, predict


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scale-run',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    scale=json.loads((args.scale_run/'comparison.json').read_text())
    scale_protocol=json.loads((args.scale_run/'protocol.json').read_text())
    assert digest(args.scale_run/'protocol.json')==scale['protocol_sha256']
    assert digest(scale_protocol['checkpoint'])==scale_protocol['checkpoint_sha256']
    method=scale['video_pilot_method'];out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    config={'source':str(args.source.resolve()),'source_sha256':digest(args.source),
            'timestamps':[round(6+i*.2,3) for i in range(15)],'fps':5,
            'roi':[0,960,2160,2880],'method':method,'confidence':.035,
            'min_hits':3,'max_missing':2,'match_iou':.3,'camera_motion':'sparse optical flow and guarded partial affine',
            'checkpoint_sha256':scale_protocol['checkpoint_sha256'],
            'scale_comparison_sha256':digest(args.scale_run/'comparison.json'),
            'script_sha256':digest(Path(__file__)),
            'helper_sha256':{name:digest(Path(__file__).resolve().parents[1]/'src/shelf_vision'/name) for name in ['scale_video.py','evaluate.py','media.py','extract_sdr.swift']},
            'versions':{name:importlib.metadata.version(name) for name in ['torch','ultralytics','numpy','pillow','opencv-python']},
            'device':device,
            'timing_scope':'Per-frame camera motion, detector call, association, synchronization; excludes decoding, crop/gray preparation, rendering, representative selection. First sample includes cold predictor initialization. Not end-to-end video throughput.',
            'warning':'Feasibility demo without persistent-tray-ID truth. Track counts are not inventory or accuracy. Green observed; yellow carried; gray tentative.'}
    save_json(out/'protocol.json',config)
    frames=[]
    for i,t in enumerate(config['timestamps']):
        path=out/'frames'/f'{i:03}.png'
        meta=extract_frame(args.source,t,path)
        assert (meta['width'],meta['height'])==(2160,3840)
        frames.append({'path':str(path),'sha256':digest(path),**meta})
    save_json(out/'frames.json',{'frames':frames})
    model=YOLO(scale_protocol['checkpoint'])
    tracker=Tracker(min_hits=3,max_missing=2,match_iou=.3)
    previous=None;records=[];renders=[];best={}
    for i,f in enumerate(frames):
        with Image.open(f['path']) as source:
            crop=source.convert('RGB').crop(config['roi'])
        small=crop.resize((720,640));gray=cv2.cvtColor(np.array(small),cv2.COLOR_RGB2GRAY)
        start=time.perf_counter()
        affine,motion=camera_motion(previous,gray)
        affine[0][2]*=3;affine[1][2]*=3
        detections=predict(model,crop,method,device)
        tracks=tracker.update(detections,affine)
        if device=='mps':torch.mps.synchronize()
        seconds=time.perf_counter()-start
        records.append({'index':i,'seconds':f['actual_seconds'],'raw_detections':detections,
                        'tracks':tracks,'motion':motion,'affine':affine,'processing_seconds':seconds,'cold_predictor':i==0})
        previous=gray
        left=small.copy();right=small.copy();ld=ImageDraw.Draw(left);rd=ImageDraw.Draw(right)
        for d in detections:ld.rectangle([v/3 for v in d['bbox_xyxy']],outline='#ff9a40',width=2)
        for tr in tracks:
            b=tr['bbox_xyxy'];clipped=[max(0,b[0]),max(0,b[1]),min(crop.width,b[2]),min(crop.height,b[3])]
            if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:continue
            if not tr['confirmed']:color='#aaaaaa'
            elif tr['observed']:color='#45ff9b'
            else:color='#ffd34e'
            box=[v/3 for v in clipped];rd.rectangle(box,outline=color,width=2)
            state='observed' if tr['observed'] else 'carried'
            rd.text((box[0]+2,box[1]+2),f"{tr['id']} {state}",fill=color,stroke_width=1,stroke_fill='black')
            if tr['observed']:
                patch=crop.crop(tuple(round(v) for v in clipped))
                if patch.width and patch.height:
                    sharp=float(cv2.Laplacian(cv2.cvtColor(np.array(patch.resize((128,128))),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
                    if sharp>best.get(tr['id'],{}).get('sharpness',-1):
                        best[tr['id']]={'sharpness':sharp,'frame_index':i,'bbox_xyxy':clipped,'confidence':tr['confidence']}
                        folder=out/'representatives';folder.mkdir(exist_ok=True);patch.thumbnail((500,500));patch.save(folder/f"track-{tr['id']:03}.jpg")
        panel=Image.new('RGB',(1440,690),'#14251e');panel.paste(left,(0,40));panel.paste(right,(720,40));draw=ImageDraw.Draw(panel)
        draw.text((10,10),f"Detector alone | {f['actual_seconds']:.2f}s",fill='white')
        draw.text((730,10),'Track IDs | green observed | yellow carried | gray tentative',fill='white')
        panel.thumbnail((1200,600));panel.save(out/f'preview-{i:03}.jpg',quality=90);renders.append(panel)
        print(i,len(detections),'detections',len(tracks),'tracks',motion['status'],flush=True)
    renders[0].save(out/'tracking.gif',save_all=True,append_images=renders[1:],duration=200,loop=0)
    confirmed=sorted({t['id'] for r in records for t in r['tracks'] if t['confirmed']})
    save_json(out/'tracks.json',{'protocol_sha256':digest(out/'protocol.json'),'frames':records,
                               'frames_manifest_sha256':digest(out/'frames.json'),
                               'representatives':best,'confirmed_track_ids':confirmed,
                               'unique_tray_recall':None,'identity_switches':None,
                               'evaluation_status':'persistent-ID ground truth required; track count is not tray count'})
    save_json(out/'identity-review-template.json',{'status':'unreviewed','instructions':'Assign the same physical-tray ID across frames. Mark all visible trays, including detector misses. Human-confirm before reporting tracking accuracy.','frames':[{'frame_index':i,'actual_seconds':f['actual_seconds'],'path':f['path'],'roi':config['roi'],'annotations':[]} for i,f in enumerate(frames)]})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Video tracking pilot</title><style>body{font:16px system-ui;margin:30px;background:#14251e;color:white}img{max-width:100%}</style><h1>Three-second tracking feasibility pilot</h1><p>Orange: raw detections. Green: observed confirmed tracks. Yellow: carried estimates. Gray: tentative. IDs can switch. No measured accuracy improvement or inventory count.</p><img src="tracking.gif"><p>Representative crops and full track history are saved beside this page. Persistent tray-ID labels are required for evaluation.</p>')
    assert digest(args.source)==config['source_sha256']
    print('Confirmed track IDs (NOT verified trays):',len(confirmed))


if __name__=='__main__':main()
