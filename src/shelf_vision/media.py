"""Read-only source inventory and explicit, reproducible media derivatives."""
import hashlib
import io
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageCms, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
IMAGE_EXTS = {'.heic', '.heif', '.jpg', '.jpeg', '.png'}
VIDEO_EXTS = {'.mov', '.mp4'}


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(source):
    source = Path(source).resolve()
    if not source.is_dir(): raise ValueError(f'Not a source directory: {source}')
    records, errors = [], []
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p.name.startswith('.'): continue
        rec = {'path': str(p.relative_to(source)), 'source_id': digest(p),
               'size_bytes': p.stat().st_size, 'status': 'ok'}
        try:
            if p.suffix.lower() in IMAGE_EXTS:
                with Image.open(p) as im:
                    im.load()
                    upright = ImageOps.exif_transpose(im)
                    rec.update(kind='image', width=upright.width, height=upright.height,
                               format=im.format, orientation=im.getexif().get(274, 1),
                               has_icc=bool(im.info.get('icc_profile')))
            elif p.suffix.lower() in VIDEO_EXTS:
                d = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)]))
                v = next(s for s in d['streams'] if s['codec_type'] == 'video')
                rotation = next((s['rotation'] for s in v.get('side_data_list',[]) if 'rotation' in s), 0)
                rec.update(kind='video', width=v['width'], height=v['height'], rotation=rotation,
                           duration_seconds=float(d['format']['duration']), fps=v['avg_frame_rate'],
                           color_transfer=v.get('color_transfer'), color_primaries=v.get('color_primaries'),
                           color_space=v.get('color_space'), codec=v['codec_name'],
                           creation_time=d['format'].get('tags',{}).get('creation_time'))
            else: raise ValueError('Unsupported media type')
        except Exception as exc:
            rec.update(status='error', error=str(exc)); errors.append({'path': rec['path'], 'error': str(exc)})
        records.append(rec)
    return {'schema_version': 1, 'source_root': str(source), 'media': records, 'errors': errors}


def prepare_images(manifest, out):
    source = Path(manifest['source_root']).resolve(); out = Path(out).resolve()
    if out == source or source in out.parents: raise ValueError('Output must be outside source directory')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for rec in manifest['media']:
        if rec['status'] != 'ok' or rec['kind'] != 'image': continue
        original = source / rec['path']
        if digest(original) != rec['source_id']: raise ValueError(f'Source changed: {rec["path"]}')
        target = out / (rec['source_id'] + '.jpg')
        with Image.open(original) as im:
            im = ImageOps.exif_transpose(im)
            icc = im.info.get('icc_profile')
            if icc:
                im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                              ImageCms.createProfile('sRGB'), outputMode='RGB')
            else: im = im.convert('RGB')
            # Fresh image strips inherited EXIF/GPS; write an explicit sRGB ICC profile.
            clean = Image.frombytes('RGB', im.size, im.tobytes())
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
            clean.save(target, quality=95, icc_profile=profile)
            records.append({'source_id': rec['source_id'], 'source_path': rec['path'],
                            'path': str(target), 'width': im.width, 'height': im.height,
                            'recipe': 'pillow-heif SDR decode; EXIF transpose; ICC to sRGB; JPEG95'})
    result = {'schema_version': 1, 'images': records}
    save_json(out / 'images.json', result)
    return result


def extract_frame(source, timestamp, out):
    if timestamp < 0: raise ValueError('timestamp must be nonnegative')
    source, out = Path(source).resolve(), Path(out).resolve()
    if out == source: raise ValueError('Output cannot replace source')
    out.parent.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).with_name('extract_sdr.swift')
    result = subprocess.run(['swift',str(helper),str(source),str(timestamp),str(out)],
                            capture_output=True, text=True, check=True)
    metadata = json.loads(result.stdout)
    metadata['recipe'] = 'AVAssetImageGenerator forceSDR; preferredTransform; CoreImage sRGB RGBA8 PNG'
    return metadata
