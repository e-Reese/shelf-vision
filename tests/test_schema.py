import pytest
from shelf_vision.schema import validate_annotation, validate_splits


def valid_area(**changes):
    a={'bbox_xyxy':[0,0,10,20],'class_name':'display_area','occupancy':'occupied','identity_state':'known','observed_sku_id':'sku-1','intended_sku_id':None}
    a.update(changes); return a


def test_empty_tray_cannot_prove_product_presence():
    validate_annotation(valid_area(), {'sku-1'})
    with pytest.raises(ValueError): validate_annotation(valid_area(occupancy='empty'),{'sku-1'})
    validate_annotation(valid_area(occupancy='empty',identity_state='not_applicable',observed_sku_id=None,intended_sku_id='sku-1'),{'sku-1'})


@pytest.mark.parametrize('changes',[{'identity_state':'unknown'}, {'observed_sku_id':'bad'}, {'bbox_xyxy':[2,0,1,20]}, {'occupancy':'mixed'}, {'occupancy':'unclear'}])
def test_inconsistent_annotations_rejected(changes):
    with pytest.raises(ValueError): validate_annotation(valid_area(**changes),{'sku-1'})


def test_one_video_cannot_cross_splits():
    with pytest.raises(ValueError): validate_splits([{'video_id':'a','split':'development'},{'video_id':'a','split':'validation'}])
    validate_splits([{'video_id':'a','split':'development'},{'video_id':'b','split':'validation'}])
