import subprocess,sys,json
from PIL import Image


def test_inventory_cli_writes_manifest_and_preserves_source(tmp_path):
    src=tmp_path/'raw';src.mkdir();p=src/'a.JPG';Image.new('RGB',(10,20)).save(p)
    original=p.read_bytes();out=tmp_path/'manifest.json'
    result=subprocess.run([sys.executable,'-m','shelf_vision.cli','inventory','--source',str(src),'--out',str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(out.read_text())['media'][0]['width']==10
    assert p.read_bytes()==original
