from pathlib import Path
import pytest
from PIL import Image
from shelf_vision.media import inventory, prepare_images, extract_frame


def test_inventory_keeps_duplicate_sources_and_reports_corruption(tmp_path):
    source = tmp_path / 'raw'; source.mkdir()
    Image.new('RGB', (20, 10), 'red').save(source / 'a.JPG')
    (source / 'b.jpg').write_bytes((source / 'a.JPG').read_bytes())
    (source / 'broken.HEIC').write_bytes(b'not an image')
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = inventory(source)
    assert len(result['media']) == 3
    good = [m for m in result['media'] if m['status'] == 'ok']
    assert len(good) == 2
    assert good[0]['source_id'] == good[1]['source_id']
    assert result['errors'][0]['path'] == 'broken.HEIC'
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


def test_prepare_applies_orientation_once_and_strips_metadata(tmp_path):
    source = tmp_path / 'raw'; source.mkdir()
    im = Image.new('RGB', (40, 20), 'red')
    for x in range(20, 40):
        for y in range(20): im.putpixel((x,y), (0,0,255))
    exif = Image.Exif(); exif[274] = 6
    im.save(source / 'rotated.jpg', exif=exif)
    manifest = inventory(source)
    prepared = prepare_images(manifest, tmp_path / 'derived')
    result = Image.open(prepared['images'][0]['path'])
    assert result.size == (20, 40)
    assert result.getexif().get(274, 1) == 1
    assert result.getpixel((10,5))[0] > 200
    assert result.getpixel((10,35))[2] > 200
    with pytest.raises(ValueError, match='source'):
        prepare_images(manifest, source / 'derived')


def test_frame_rejects_negative_timestamp(tmp_path):
    with pytest.raises(ValueError, match='timestamp'):
        extract_frame(Path('missing.mov'), -1, tmp_path / 'out.png')
