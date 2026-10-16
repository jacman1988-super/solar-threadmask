import hashlib
import numpy as np
import pytest
import runtime
import json

@pytest.mark.parametrize('value',[-1,256,.5,np.nan,np.inf])
def test_no_silent_pixel_wrap_or_truncation(value):
    image=np.zeros((2048,2048),np.float32);image[0,0]=value
    with pytest.raises(ValueError):runtime.prepare(image)

def test_unknown_method_is_rejected_before_loading_models():
    with pytest.raises(ValueError,match='Unknown method'):runtime.segment(np.zeros((8,8)), '../model')

def test_weight_change_invalidates_loaded_model(tmp_path,monkeypatch):
    path=tmp_path/'model.joblib';path.write_bytes(b'first');digest=hashlib.sha256(b'first').hexdigest()
    monkeypatch.setattr(runtime.joblib,'load',lambda path:object())
    runtime._forest.cache_clear();runtime._forest(path,digest)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='integrity'):runtime.verified_artifact(path,digest)
    second=hashlib.sha256(b'changed').hexdigest()
    assert runtime._forest(path,second) is not runtime._forest(path,digest)

def test_image_only_mask_matches_frozen_actual_image(monkeypatch):
    import research
    from pipeline import load_image
    expected=research.read(research.HERE/'integration-checks.json')
    def forbidden(*args,**kwargs):raise AssertionError('Used annotations or stale image cache')
    monkeypatch.setattr(research.Dataset,'__init__',forbidden)
    monkeypatch.setattr(research,'probabilities',forbidden)
    image=load_image(research.DATA/'train/train_images'/expected['file'])
    labels=runtime.segment(image,'rf_filter')
    assert int(labels.max())==expected['instances']
    assert hashlib.sha256(labels.tobytes()).hexdigest()==expected['label_array_sha256']

def test_changed_component_protocol_cannot_change_prediction_silently(tmp_path,monkeypatch):
    import shutil
    for name in ['component-frozen.json','component-protocol.json','rf-frozen.json']:
        shutil.copyfile(runtime.HERE/name,tmp_path/name)
    path=tmp_path/'component-protocol.json';protocol=json.loads(path.read_text())
    protocol['candidate']['threshold']=.99;path.write_text(json.dumps(protocol))
    monkeypatch.setattr(runtime,'HERE',tmp_path)
    monkeypatch.setattr(runtime,'prepare',lambda image:(np.zeros((4,4)),np.ones((4,4),bool)))
    monkeypatch.setattr(runtime,'verified_artifact',lambda p,d:p)
    monkeypatch.setattr(runtime,'probability_map',lambda *a:np.zeros((4,4)))
    monkeypatch.setattr(runtime,'instances',lambda *a,**k:np.zeros((2048,2048),np.int32))
    monkeypatch.setattr(runtime,'component_features',lambda *a:np.zeros((0,14)))
    monkeypatch.setattr(runtime,'_forest',lambda *a:object())
    with pytest.raises(ValueError,match='protocol'):runtime.segment(None,'rf_filter')
