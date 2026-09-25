from .catalog import validate_box


def validate_annotation(record, catalog_ids, width=10**9, height=10**9):
    validate_box(record['bbox_xyxy'], width, height)
    if record['class_name']!='display_area': raise ValueError('Only display_area is supported')
    occupancy=record['occupancy']; identity=record['identity_state']; sku=record['observed_sku_id']
    allowed={'occupied':{'known','unknown','unresolved'},'empty':{'not_applicable'},'mixed':{'not_applicable'},'unclear':{'unresolved'}}
    if occupancy not in allowed or identity not in allowed[occupancy]: raise ValueError('Inconsistent occupancy and identity')
    if identity=='known':
        if sku not in catalog_ids: raise ValueError('Unknown catalog SKU')
    elif sku is not None: raise ValueError('Only known observed products may have an observed SKU')
    intended=record.get('intended_sku_id')
    if intended is not None and intended not in catalog_ids: raise ValueError('Invalid intended SKU')


def validate_splits(frames):
    splits={}
    for frame in frames:
        if frame['split'] not in {'development','validation','test'}: raise ValueError('Invalid split')
        previous=splits.setdefault(frame['video_id'],frame['split'])
        if previous!=frame['split']: raise ValueError('A video cannot appear in multiple splits')
