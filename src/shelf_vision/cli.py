"""Local command-line entry points; model dependencies load only when needed."""
import argparse
import json
from pathlib import Path
from .media import inventory, prepare_images, save_json


def main():
    parser=argparse.ArgumentParser(prog='shelf-vision')
    sub=parser.add_subparsers(dest='command',required=True)
    specs={
      'inventory':['source','out'], 'prepare':['manifest','out'],
      'catalog-sheet':['media','out'], 'catalog-validate':['catalog'],
      'pilot-sheet':['frames','out'], 'annotations-validate':['data'],
      'baseline':['config','data','out'], 'evaluate':['data','predictions','out'],
      'deployment-probe':['data','out']}
    for name,options in specs.items():
        command=sub.add_parser(name)
        for option in options:command.add_argument('--'+option,required=True)
    args=parser.parse_args()
    load=lambda path:json.loads(Path(path).read_text())
    if args.command=='inventory':
        result=inventory(args.source);save_json(args.out,result)
        print(f"Inventoried {len(result['media'])} sources; {len(result['errors'])} errors")
        if result['errors']:raise SystemExit(1)
    elif args.command=='prepare':
        result=prepare_images(load(args.manifest),args.out);print(f"Prepared {len(result['images'])} images")
    elif args.command=='catalog-sheet':
        from .catalog import contact_sheet
        # Resolve prepared images from their source IDs; originals remain read-only.
        manifest=load(args.media);images=load('data/derived/photos/images.json')['images'];by_id={i['source_id']:i for i in images}
        contact_sheet([{'path':by_id[r['source_id']]['path'],'label':r['path']} for r in manifest['media'] if r.get('kind')=='image' and r['status']=='ok'],args.out)
    elif args.command=='catalog-validate':
        from .catalog import validate_catalog
        images=load('data/derived/photos/images.json')['images']
        print(json.dumps(validate_catalog(load(args.catalog),{i['source_id']:i for i in images}),indent=2))
    elif args.command=='pilot-sheet':
        from .catalog import contact_sheet
        contact_sheet([{'path':f['path'],'label':f["frame_id"]+' '+f['split']} for f in load(args.frames)['frames']],args.out)
    elif args.command=='annotations-validate':
        from .schema import validate_annotation,validate_splits
        data=Path(args.data);frames=load(data/'frames.json')['frames'];validate_splits(frames)
        by_id={f['frame_id']:f for f in frames};ids={p['sku_id'] for p in load(data/'catalog.json')['products']}
        labels=load(data/'annotations.json');rois={r['roi_id']:r for r in labels['rois']}
        for a in labels['annotations']:
            f=by_id[a['frame_id']]
            if rois[a['review_roi_id']]['frame_id']!=f['frame_id']:raise ValueError('ROI/frame mismatch')
            validate_annotation(a,ids,f['width'],f['height'])
        print(f"Validated {len(labels['annotations'])} annotations; human review: {labels.get('human_review_status','unknown')}")
    elif args.command=='baseline':
        from .baseline import run_baseline
        run_baseline(args.config,args.data,args.out)
    elif args.command=='evaluate':
        from .evaluate import evaluate
        result=evaluate(args.data,args.predictions,args.out);print(json.dumps(result['splits'],indent=2))
    elif args.command=='deployment-probe':
        from .deployment_probe import deployment_probe
        print(json.dumps(deployment_probe(args.data,args.out),indent=2))

if __name__=='__main__':main()
