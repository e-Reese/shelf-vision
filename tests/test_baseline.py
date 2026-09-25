import numpy as np
import pytest
from shelf_vision.baseline import rank_skus, cache_key


def test_ranking_normalizes_and_returns_distinct_skus():
    result=rank_skus(np.array([2.,0.]),np.array([[10.,0.],[1.,1.],[0.,1.]]),['a','a','b'])
    assert [x['sku_id'] for x in result]==['a','b']
    assert result[0]['score']==pytest.approx(1.)
    assert result[1]['score']==pytest.approx(0.)
    with pytest.raises(ValueError): rank_skus(np.array([1.,0.]),np.empty((0,2)),[])
    with pytest.raises(ValueError): rank_skus(np.zeros(2),np.ones((1,2)),['a'])


def test_cache_invalidates_configuration_changes():
    base={'source':'a','crop':[0,0,10,10],'gallery':'g1','prompt':'tray','revision':'r1'}
    for field,value in [('crop',[1,0,10,10]),('gallery','g2'),('prompt','box'),('revision','r2')]:
        assert cache_key(base)!=cache_key({**base,field:value})


def test_cached_predictions_follow_current_frame_identity(tmp_path):
    import json
    from shelf_vision.baseline import load_detection_cache
    path=tmp_path/'cache.json'
    path.write_text(json.dumps({'frame_id':'old','boxes':[{'bbox_xyxy':[0,0,5,5]}]}))
    result=load_detection_cache(path,'new')
    assert result['frame_id']=='new'
    assert result['boxes'][0]['bbox_xyxy']==[0,0,5,5]
    assert json.loads(path.read_text())['frame_id']=='old'
