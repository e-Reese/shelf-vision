"""Check exported raw outputs using exactly the same square RGB input."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import coremltools as ct
from ultralytics import YOLO
from shelf_vision.media import save_json
out=Path('reports/phase-0/deployment');r=json.loads((out/'result.json').read_text())
model=YOLO(r['checkpoint']);start=time.perf_counter()
package=model.export(format='coreml',imgsz=640,nms=False,half=False)
converted=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
original=YOLO(r['checkpoint']).model.cpu().eval()
results=[]
for path in sorted((Path(r.get('dataset',str(out/'dataset')))/'images/val').glob('*.jpg'))[:3]:
    image=Image.open(path).convert('RGB').resize((640,640),Image.Resampling.BILINEAR)
    tensor=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).unsqueeze(0).float()/255
    with torch.inference_mode(): native=original(tensor)[0].numpy()
    prediction=converted.predict({'image':image})
    exported=next(iter(prediction.values()))
    if native.shape!=exported.shape: raise ValueError(f'Output shape mismatch: {native.shape} vs {exported.shape}')
    difference=np.abs(native-exported)
    results.append({'image':path.name,'shape':list(native.shape),
                    'max_abs_box_error_pixels':float(difference[:,:4].max()),
                    'max_abs_score_error':float(difference[:,4:].max()),
                    'native_max_score':float(native[:,4:].max()),
                    'finite':bool(np.isfinite(exported).all()),
                    'passed':bool(np.isfinite(exported).all() and difference[:,:4].max()<=1.0 and difference[:,4:].max()<=.001)})
r['raw_parity']=results;r['raw_parity_policy']={'identical_input':'640x640 RGB bilinear resize; native /255 vs CoreML image preprocessing','compute_units':'CPU_ONLY','max_box_error_pixels':1.0,'max_score_error':.001}
r['coreml_package']=str(package);r['raw_parity_seconds']=time.perf_counter()-start
r['export_status']='passed_raw_parity' if all(x['passed'] for x in results) else 'raw_parity_failed'
save_json(out/'result.json',r);print(json.dumps(r,indent=2))
