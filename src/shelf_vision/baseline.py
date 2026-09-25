"""Local detector and image retrieval baseline, with explicit provenance."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from .media import save_json,digest


def cache_key(settings):
    return hashlib.sha256(json.dumps(settings,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_detection_cache(path, frame_id):
    record = json.loads(Path(path).read_text())
    record['frame_id'] = frame_id
    record['cache_hit'] = True
    return record


def rank_skus(query,gallery,sku_ids,k=3):
    query=np.asarray(query,dtype=float); gallery=np.asarray(gallery,dtype=float)
    if not len(sku_ids) or len(gallery)!=len(sku_ids): raise ValueError('Empty or mismatched gallery')
    qn=np.linalg.norm(query); gn=np.linalg.norm(gallery,axis=1)
    if not np.isfinite(query).all() or not np.isfinite(gallery).all() or qn==0 or np.any(gn==0): raise ValueError('Invalid embedding')
    scores=gallery@query/(gn*qn); best={}
    for sku,score in zip(sku_ids,scores): best[sku]=max(best.get(sku,-float('inf')),float(score))
    return [{'sku_id':sku,'score':score} for sku,score in sorted(best.items(),key=lambda x:(-x[1],x[0]))[:k]]


def run_baseline(config_path,data_dir,out,gallery_dir=None):
    import torch
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,AutoImageProcessor,AutoModel
    config=json.loads(Path(config_path).read_text()); data_dir=Path(data_dir); out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frames=json.loads((data_dir/'frames.json').read_text())['frames']
    catalog=json.loads((data_dir/'catalog.json').read_text())['products']
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    def sync():
        if device=='mps': torch.mps.synchronize()
    start=time.perf_counter()
    model_id=config['detector']; revision=config.get('detector_revision','main')
    processor=AutoProcessor.from_pretrained(model_id,revision=revision)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(model_id,revision=revision).to(device).eval()
    detector.config.disable_custom_kernels=True
    detection_revision=detector.config._commit_hash
    load_seconds=time.perf_counter()-start
    detections=[]
    for frame in frames:
        image=Image.open(frame['path']).convert('RGB')
        settings={'source':digest(frame['path']),'model':model_id,'revision':detection_revision,
                  'prompt':config['prompt'],'box_threshold':config['box_threshold'],'text_threshold':config['text_threshold'],
                  'processor':processor.to_dict() if hasattr(processor,'to_dict') else str(processor), 'device':device}
        key=cache_key(settings); cached=out/(key+'.json')
        if cached.exists(): rec=load_detection_cache(cached,frame['frame_id'])
        else:
            inputs=processor(images=image,text=config['prompt'],return_tensors='pt').to(device)
            sync(); t=time.perf_counter()
            with torch.inference_mode(): outputs=detector(**inputs)
            sync(); seconds=time.perf_counter()-t
            result=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
                threshold=config['box_threshold'],text_threshold=config['text_threshold'],target_sizes=[image.size[::-1]])[0]
            rec={'frame_id':frame['frame_id'],'cache_hit':False,'inference_seconds':seconds,
                 'boxes':[{'bbox_xyxy':box,'confidence':score,'label':label} for box,score,label in
                    zip(result['boxes'].cpu().tolist(),result['scores'].cpu().tolist(),result.get('text_labels',result['labels']))],
                 'provenance':settings}
            save_json(cached,rec)
        detections.append(rec)
        print(frame['frame_id'],len(rec['boxes']),'boxes',round(rec['inference_seconds'],2),'s',flush=True)
    del detector
    if device=='mps': torch.mps.empty_cache()
    embed_start=time.perf_counter()
    encoder_id=config['encoder']; revision=config.get('encoder_revision','main')
    image_processor=AutoImageProcessor.from_pretrained(encoder_id,revision=revision)
    encoder=AutoModel.from_pretrained(encoder_id,revision=revision).to(device).eval()
    encoder_revision=encoder.config._commit_hash
    encoder_load_seconds=time.perf_counter()-embed_start
    def embed(images):
        # Square padding preserves complete, often elongated, package fronts.
        images=[ImageOps.pad(im.convert('RGB'),(224,224),color=(127,127,127)) for im in images]
        inputs=image_processor(images=images,return_tensors='pt',do_center_crop=False).to(device)
        with torch.inference_mode(): vectors=encoder(**inputs).last_hidden_state[:,0]
        return vectors.cpu().numpy()
    gallery_paths=[Path(gallery_dir or 'data/derived/catalog')/(p['sku_id']+'.jpg') for p in catalog]
    gallery=embed([Image.open(p) for p in gallery_paths]); ids=[p['sku_id'] for p in catalog]
    np.savez(out/'gallery.npz',vectors=gallery,sku_ids=np.array(ids))
    embedding_provenance={'encoder':encoder_id,'revision':encoder_revision,'gallery_hash':cache_key([digest(p) for p in gallery_paths]),
                          'preprocessing':'RGB; aspect-preserving square pad 224 gray127; processor normalization; CLS embedding',
                          'aggregation':'maximum cosine similarity per SKU','device':device}
    for frame,rec in zip(frames,detections):
        image=Image.open(frame['path']).convert('RGB')
        boxes=rec['boxes']; sync(); t=time.perf_counter()
        for i in range(0,len(boxes),16):
            batch=boxes[i:i+16]
            vectors=embed([image.crop(b['bbox_xyxy']) for b in batch])
            for b,vector in zip(batch,vectors): b['matches']=rank_skus(vector,gallery,ids)
        sync(); rec['retrieval_seconds']=time.perf_counter()-t
    # Optional verified-area diagnostic. Labels retain their review provenance.
    annotations_path=data_dir/'annotations.json'
    crop_results=[]
    if annotations_path.exists():
        annotations=json.loads(annotations_path.read_text())['annotations'];by_frame={f['frame_id']:f for f in frames}
        for a in annotations:
            if a['occupancy']!='occupied':continue
            image=Image.open(by_frame[a['frame_id']]['path']).convert('RGB')
            for kind,box in [('area',a['bbox_xyxy']),('package',a.get('package_bbox_xyxy'))]:
                if box:
                    crop_results.append({'region_id':a['region_id'],'crop_kind':kind,'matches':rank_skus(embed([image.crop(box)])[0],gallery,ids)})
    save_json(out/'predictions.json',{'schema_version':1,'config':config,'device':device,'detector_load_seconds':load_seconds,
              'encoder_load_seconds':encoder_load_seconds,'embedding_provenance':embedding_provenance,'frames':detections,
              'crop_results':crop_results,'total_seconds':time.perf_counter()-start})
    print('Saved',out/'predictions.json',flush=True)
