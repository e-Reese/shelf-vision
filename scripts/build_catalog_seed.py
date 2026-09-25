"""Materialize visually reviewed photo groups; retain explicit identity uncertainty."""
import json
from pathlib import Path
from PIL import Image
from shelf_vision.media import save_json
from shelf_vision.catalog import validate_catalog, contact_sheet, raster_sheet
records=json.loads(Path('data/derived/photos/images.json').read_text())['images']
by_number={int(Path(r['source_path']).stem.split('_')[1]):r for r in records}
seeds=json.loads(Path('configs/catalog-seed.json').read_text())
products=[]; cards=[]
for index,(number,name,box) in enumerate(seeds,1):
    sku=f'sku-{index:04}'
    refs=[]
    for offset,view in enumerate(['front','side','back']):
        photo_number = 3434 if number == 3431 and offset == 2 else number + offset
        record=by_number[photo_number]
        crop=[round(box[i]*[record['width'],record['height']][i%2]) for i in range(4)] if offset==0 else None
        refs.append({'source_id':record['source_id'],'source_path':record['source_path'],'view':view,'bbox_xyxy':crop})
        if offset==0:
            target=Path('data/derived/catalog')/(sku+'.jpg'); target.parent.mkdir(parents=True,exist_ok=True)
            Image.open(record['path']).crop(crop).save(target,quality=95)
            cards.append({'path':str(target),'label':f'{sku}: {name}'})
    if number==3436:
        record=by_number[3435]
        refs.append({'source_id':record['source_id'],'source_path':record['source_path'],'view':'front','bbox_xyxy':None,'note':'Additional in-shelf view, excluded from reference gallery pending crop review'})
    products.append({'sku_id':sku,'name':name,'size':None,'barcode':None,'status':'agent_verified',
                     'human_review_status':'pending','references':refs,'identity_note':'Name/variant visually reviewed; size and barcode not transcribed'})
catalog={'schema_version':1,'products':products}
save_json('data/manifests/catalog.json',catalog)
print(validate_catalog(catalog,{r['source_id']:r for r in records}))
contact_sheet(cards,'reports/phase-0/catalog-crops.html')
for n in range(0,40,20): raster_sheet(cards[n:n+20],f'reports/phase-0/crops-{n//20}.jpg',5)
