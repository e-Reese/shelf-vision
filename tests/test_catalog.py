import pytest
from shelf_vision.catalog import validate_catalog, contact_sheet


def entry(sku, ref='a', box=None):
    return {'sku_id':sku, 'status':'agent_verified', 'name':sku,
            'references':[{'source_id':ref, 'view':'front', 'bbox_xyxy':box or [0,0,10,20]}]}


def test_duplicate_refs_and_invalid_crops_rejected():
    images = {'a': {'width':10, 'height':20}}
    assert validate_catalog({'products':[entry('sku-1')]},images)['products']==1
    with pytest.raises(ValueError, match='assigned'):
        validate_catalog({'products':[entry('sku-1'),entry('sku-2')]}, images)
    with pytest.raises(ValueError, match='bounds'):
        validate_catalog({'products':[entry('sku-1', box=[0,0,11,20])]}, images)
    with pytest.raises(ValueError, match='source'):
        validate_catalog({'products':[entry('sku-1',ref='missing')]}, images)


def test_missing_front_is_reported_and_extra_photo_does_not_shift_ids():
    e = entry('sku-19'); e['references'][0]['view']='back'
    result = validate_catalog({'products':[e]}, {'a':{'width':10,'height':20}})
    assert result['missing_front']==['sku-19']


def test_contact_sheet_escapes_labels(tmp_path):
    from PIL import Image
    p=tmp_path/'a.jpg'; Image.new('RGB',(10,20)).save(p)
    out=tmp_path/'sheet.html'
    contact_sheet([{'path':str(p),'label':'<script>alert(1)</script>'}],out)
    text=out.read_text()
    assert '<script>alert' not in text
    assert '&lt;script&gt;' in text
    assert 'a.jpg' in text
