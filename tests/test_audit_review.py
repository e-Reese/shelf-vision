import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from shelf_vision.editor_store import EditorStore
from shelf_vision.editor_api import create_app
from shelf_vision.media import digest


@pytest.fixture
def audit_client(tmp_path):
    audit=tmp_path/'reports/phase-2/tray-audit-002';audit.mkdir(parents=True)
    snap=tmp_path/'reports/phase-2/visit-002-evaluation/run-001/manifests';snap.mkdir(parents=True)
    image=tmp_path/'crop.png';Image.new('RGB',(100,80)).save(image)
    annotations={'annotations':[{'region_id':'r1','frame_id':'f1','bbox_xyxy':[10,10,40,40]},
                                {'region_id':'r2','frame_id':'f1','bbox_xyxy':[50,10,80,40]}]}
    (snap/'annotations.json').write_text(json.dumps(annotations))
    (snap/'frames.json').write_text(json.dumps({'frames':[{'frame_id':'f1','path':str(image),'width':100,'height':80,'crop_sha256':digest(image)}]}))
    (audit/'audit.json').write_text(json.dumps({'source_annotations_sha256':digest(snap/'annotations.json'),'decisions':[
        {'region_id':'r1','frame_id':'f1','number':1,'status':'boundary_or_tray_presence_uncertain','reason':'Check tray'},
        {'region_id':'r2','frame_id':'f1','number':2,'status':'exclude_user_confirmed_nontray','reason':'User confirmed','reviewer':'Pascal'}]}))
    store=EditorStore(tmp_path/'editor.sqlite',[])
    with TestClient(create_app(store,root=tmp_path)) as client:
        yield client,tmp_path


def test_review_saves_and_reloads_without_changing_source_audit(audit_client):
    client,root=audit_client
    path=root/'reports/phase-2/tray-audit-002/audit.json';before=path.read_bytes()
    data=client.get('/api/audit').json()
    assert data['items'][0]['decision']=='pending'
    assert data['items'][1]['decision']=='not_tray'
    saved=client.put('/api/audit/r1',json={'revision':0,'decision':'tray','note':'Visible carton'}).json()
    assert saved['revision']==1 and saved['decision']=='tray'
    assert client.get('/api/audit').json()['items'][0]['note']=='Visible carton'
    assert path.read_bytes()==before
    assert client.get('/api/audit/frames/f1/image').status_code==200
    page=client.get('/audit')
    assert page.status_code==200 and 'Tray review' in page.text


def test_stale_review_cannot_overwrite_new_decision(audit_client):
    client,_=audit_client
    assert client.put('/api/audit/r1',json={'revision':0,'decision':'tray','note':''}).status_code==200
    assert client.put('/api/audit/r1',json={'revision':0,'decision':'not_tray','note':''}).status_code==409
    assert client.get('/api/audit').json()['items'][0]['decision']=='tray'


def test_invalid_or_unknown_decision_rejected(audit_client):
    client,_=audit_client
    assert client.put('/api/audit/r1',json={'revision':0,'decision':'banana','note':''}).status_code==422
    assert client.put('/api/audit/missing',json={'revision':0,'decision':'tray','note':''}).status_code==404
    assert client.get('/api/audit/frames/missing/image').status_code==404
    assert client.get('/api/audit').json()['items'][0]['revision']==0


def test_export_contains_decisions_source_identity_and_history(audit_client):
    client,_=audit_client
    client.put('/api/audit/r1',json={'revision':0,'decision':'needs_adjustment','note':'Right edge too wide'})
    data=client.get('/api/audit/export').json()
    assert data['items'][0]['decision']=='needs_adjustment'
    assert data['events'][0]['before']['decision']=='pending'
    assert data['events'][0]['after']['decision']=='needs_adjustment'
    assert data['source_audit_sha256']


def test_changed_source_refused_after_review_initialization(audit_client):
    client,root=audit_client
    client.get('/api/audit')
    p=root/'reports/phase-2/tray-audit-002/audit.json'
    p.write_text(p.read_text()+' ')
    assert client.get('/api/audit').status_code==422


def test_training_review_isolated_from_visit_review(audit_client):
    import shutil
    client,root=audit_client
    source=root/'reports/phase-2/visit-002-evaluation/run-001/manifests'
    target=root/'reports/phase-2/training-tray-audit-001'
    shutil.copytree(source,target/'manifests')
    shutil.copy2(root/'reports/phase-2/tray-audit-002/audit.json',target/'audit.json')
    response=client.get('/api/audit?review=training')
    assert response.status_code==200
    assert response.json()['review']=='training'
    assert client.put('/api/audit/r1?review=training',json={'revision':0,'decision':'tray','note':'training only'}).status_code==200
    assert client.get('/api/audit?review=training').json()['items'][0]['decision']=='tray'
    assert client.get('/api/audit').json()['items'][0]['decision']=='pending'
    image_url=response.json()['frames'][0]['image_url']
    assert image_url.endswith('?review=training')
    assert client.get(image_url).status_code==200
    assert client.get('/api/audit/export?review=training').json()['events'][0]['after']['note']=='training only'
    assert client.get('/api/audit?review=../../other').status_code==422
