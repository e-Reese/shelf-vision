import json
from pathlib import Path
from PIL import Image
from shelf_vision.media import extract_frame,save_json
from shelf_vision.schema import validate_splits
from shelf_vision.catalog import contact_sheet,raster_sheet
selection=json.loads(Path('configs/pilot-selection.json').read_text())
media=json.loads(Path('data/manifests/media.json').read_text())
by_path={m['path']:m for m in media['media']}
frames=[]
for video in selection['videos']:
    for t in video['timestamps']:
        fid=f'frame-{len(frames)+1:04}'
        path=Path('data/derived/frames')/(fid+'.png')
        metadata=extract_frame(Path(media['source_root'])/video['path'],t,path)
        frame={'frame_id':fid,'video_id':by_path[video['path']]['source_id'],'video_path':video['path'],
               'split':video['split'],'capture_group':selection['capture_group'],'path':str(path.resolve()),**metadata}
        frames.append(frame)
        im=Image.open(path).convert('RGB'); im.thumbnail((540,960));im.save(path.with_suffix('.jpg'),quality=95)
        print(fid,video['path'],t,flush=True)
validate_splits(frames)
save_json('data/manifests/frames.json',{'schema_version':1,'frames':frames})
items=[{'path':str(Path(f['path']).with_suffix('.jpg')),'label':f"{f['frame_id']} {f['video_path']} {f['requested_seconds']}s {f['split']}"} for f in frames]
contact_sheet(items,'reports/phase-0/pilot.html')
raster_sheet(items,'reports/phase-0/pilot.jpg',5)
