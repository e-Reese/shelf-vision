import json
from pathlib import Path
from PIL import Image,ImageDraw
from shelf_vision.media import save_json
from shelf_vision.schema import validate_annotation
from shelf_vision.catalog import contact_sheet
seed=json.loads(Path('configs/pilot-annotation-seed.json').read_text())
frames=json.loads(Path('data/manifests/frames.json').read_text())['frames']
catalog=json.loads(Path('data/manifests/catalog.json').read_text())['products']
ids={p['sku_id'] for p in catalog}; names={p['sku_id']:p['name'] for p in catalog}
annotations=[]; rois=[]; sheets=[]
for frame in frames:
    n=str(int(frame['frame_id'].split('-')[1]));im=Image.open(frame['path']).convert('RGB');im.thumbnail((540,960));draw=ImageDraw.Draw(im)
    for section in ['frames','unknown_regions']:
        if n not in seed[section]:continue
        spec=seed[section][n];rid=frame['frame_id']+'-'+section
        scale=[frame['width']/540,frame['height']/960]*2
        rois.append({'roi_id':rid,'frame_id':frame['frame_id'],'bbox_xyxy':[round(v*s) for v,s in zip(spec['roi'],scale)],'review_status':'agent_reviewed'})
        draw.rectangle(spec['roi'],outline='cyan',width=2)
        for index,area in enumerate(spec['areas']):
            identity,*box=area
            sku=f'sku-{identity:04}' if isinstance(identity,int) else None
            occupancy='unclear' if identity=='unclear' else 'occupied'
            state='known' if sku else 'unresolved' if identity in {'unclear','unresolved'} else 'unknown'
            a={'region_id':rid+f'-{index:02}','frame_id':frame['frame_id'],'review_roi_id':rid,
               'bbox_xyxy':[round(v*s) for v,s in zip(box,scale)],'class_name':'display_area','occupancy':occupancy,
               'identity_state':state,'observed_sku_id':sku,'intended_sku_id':None,'truncated':False,
               'review_status':'agent_reviewed','human_review_status':'pending'}
            package=seed.get('paired_package_crops',{}).get(n,{}).get(str(index)) if section=='frames' else None
            if package:a['package_bbox_xyxy']=[round(v*s) for v,s in zip(package,scale)]
            a['truncated']=any(box[i]==spec['roi'][i] for i in range(4))
            validate_annotation(a,ids,frame['width'],frame['height']);annotations.append(a)
            draw.rectangle(box,outline='lime' if sku else 'orange',width=2)
            draw.text((box[0]+2,box[1]+2),sku or state,fill='black',stroke_width=1,stroke_fill='white')
    out=Path('reports/phase-0/annotations')/(frame['frame_id']+'.jpg');out.parent.mkdir(parents=True,exist_ok=True);im.save(out,quality=95)
    sheets.append({'path':str(out),'label':frame['frame_id']+' candidate annotations; human review pending'})
save_json('data/manifests/annotations.json',{'schema_version':1,'review_status':'agent_reviewed','human_review_status':'pending','rois':rois,'annotations':annotations})
contact_sheet(sheets,'reports/phase-0/annotation-review.html')
print(len(annotations),'candidate areas; human review pending')
