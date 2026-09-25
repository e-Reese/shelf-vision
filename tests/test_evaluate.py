import pytest
from shelf_vision.evaluate import match_boxes,recognition_metrics


def test_duplicates_are_false_positives_and_matching_is_one_to_one():
    gt=[{'bbox_xyxy':[0,0,10,10]},{'bbox_xyxy':[20,0,30,10]}]
    pred=[{'bbox_xyxy':[0,0,10,10],'confidence':.9},{'bbox_xyxy':[0,0,10,10],'confidence':.8}]
    matches,fp,fn=match_boxes(pred,gt)
    assert matches==[(0,0)]
    assert (fp,fn)==(1,1)


def test_unresolved_and_empty_are_not_unknown_or_correct_products():
    annotations=[{'region_id':'a','occupancy':'occupied','identity_state':'known','observed_sku_id':'sku-1'},
      {'region_id':'b','occupancy':'occupied','identity_state':'unknown','observed_sku_id':None},
      {'region_id':'c','occupancy':'unclear','identity_state':'unresolved','observed_sku_id':None},
      {'region_id':'d','occupancy':'empty','identity_state':'not_applicable','observed_sku_id':None,'intended_sku_id':'sku-1'}]
    crops=[{'region_id':r,'matches':[{'sku_id':'sku-1','score':.9}]} for r in ['a','b','c','d']]
    result=recognition_metrics(annotations,crops,.8)
    assert result['top1']=={'numerator':1,'denominator':1,'rate':1.0}
    assert result['unknown_false_accept']=={'numerator':1,'denominator':1,'rate':1.0}
    assert result['excluded']==2
    assert recognition_metrics([],[],.8)['top1']['rate'] is None


def test_known_rejection_is_visible_when_all_unknowns_rejected():
    a=[{'region_id':'a','occupancy':'occupied','identity_state':'known','observed_sku_id':'x'}]
    r=recognition_metrics(a,[{'region_id':'a','matches':[{'sku_id':'x','score':.5}]}],.9)
    assert r['known_acceptance']['rate']==0
