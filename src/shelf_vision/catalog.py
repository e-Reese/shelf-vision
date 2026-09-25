"""Catalog integrity and local review sheets."""
from pathlib import Path
from html import escape
import os
from urllib.parse import quote
from PIL import Image, ImageDraw


def validate_box(box, width, height):
    import math
    if len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box):
        raise ValueError('Invalid box coordinates')
    x1,y1,x2,y2=box
    if not (0<=x1<x2<=width and 0<=y1<y2<=height): raise ValueError('Box outside image bounds')


def validate_catalog(catalog, images):
    assigned=set(); ids=set(); missing=[]
    for p in catalog['products']:
        if p['sku_id'] in ids: raise ValueError('Duplicate SKU ID')
        ids.add(p['sku_id'])
        if not any(r['view']=='front' for r in p['references']): missing.append(p['sku_id'])
        for r in p['references']:
            sid=r['source_id']
            if sid not in images: raise ValueError(f'Unknown source {sid}')
            if sid in assigned: raise ValueError(f'Reference already assigned: {sid}')
            assigned.add(sid)
            if r['view'] not in {'front','side','back','barcode'}: raise ValueError('Unknown reference view')
            if r.get('bbox_xyxy') is not None:
                validate_box(r['bbox_xyxy'], images[sid]['width'], images[sid]['height'])
    return {'products':len(ids), 'assigned_photos':len(assigned), 'unassigned_photos':len(set(images)-assigned),
            'missing_front':missing}


def contact_sheet(items, out):
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    cards=[]
    for item in items:
        relative=quote(os.path.relpath(Path(item['path']).resolve(),out.parent.resolve()))
        label=escape(item['label'])
        cards.append(f'<figure><a href="{relative}"><img loading="lazy" src="{relative}"></a><figcaption>{label}</figcaption></figure>')
    out.write_text('<!doctype html><meta charset="utf-8"><title>Shelf Vision contact sheet</title>'
                   '<style>body{font:16px system-ui;background:#eee;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%;height:320px;object-fit:contain}figcaption{overflow-wrap:anywhere}</style>'
                   '<h1>Shelf Vision review sheet</h1><p>Click an image for full resolution. Candidate grouping requires review.</p><main>'+''.join(cards)+'</main>')


def raster_sheet(items,out,columns=5):
    cellw,cellh=216,404
    canvas=Image.new('RGB',(columns*cellw,((len(items)+columns-1)//columns)*cellh),'white')
    draw=ImageDraw.Draw(canvas)
    for i,item in enumerate(items):
        with Image.open(item['path']) as im:
            im=im.convert('RGB'); im.thumbnail((cellw,cellh-28))
            x=(i%columns)*cellw; y=(i//columns)*cellh
            canvas.paste(im,(x+(cellw-im.width)//2,y))
            draw.text((x+4,y+cellh-25),item['label'],fill='black')
    Path(out).parent.mkdir(parents=True,exist_ok=True); canvas.save(out)
