import itertools,json
import numpy as np
import pytest
from research import HERE,instances,pixel_features,channels

def test_dates_do_not_leak_between_roles():
    splits=json.loads((HERE/'protocol.json').read_text())['split']
    for a,b in itertools.combinations(splits,2):
        assert not {n[:8] for n in splits[a]} & {n[:8] for n in splits[b]}

def test_empty_prediction_stays_empty():
    out=instances(np.zeros((64,64)),np.ones((64,64),bool),.5,4,0)
    assert out.shape==(2048,2048) and not out.any()

def test_objects_are_distinct_and_small_noise_removed():
    p=np.zeros((64,64));p[20:28,20:28]=1;p[35:43,35:43]=1;p[30,30]=1
    lab=instances(p,np.ones(p.shape,bool),.5,8,0)
    assert set(np.unique(lab))=={0,1,2}
    assert (lab>0).sum()==128*32*32

def test_limb_excluded_without_using_annotations():
    disk=np.zeros((64,64),bool);disk[8:56,8:56]=1
    p=np.zeros((64,64));p[9:12,20:25]=1
    assert not instances(p,disk,.5,1,0).any()

def test_features_finite_and_reproducible_on_dark_input():
    im=np.zeros((64,64),np.float32);disk=np.zeros(im.shape,bool)
    a=pixel_features(im,disk);b=pixel_features(im,disk)
    assert a.shape==(64,64,24) and np.isfinite(a).all()
    np.testing.assert_array_equal(a,b)

def test_unet_restores_odd_spatial_shape_and_backpropagates():
    import torch
    from neural import Net
    torch.set_num_threads(2);model=Net();x=torch.zeros(1,3,65,67)
    y=model(x);assert y.shape==(1,1,65,67)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_cold_and_warm_cache_produce_same_boundary_decisions(tmp_path,monkeypatch):
    import research,joblib
    class Data:
        def arrays(self,name):return np.zeros((16,16)),None,np.ones((16,16),bool)
    class Model:
        def predict(self,x):return np.linspace(.3999,.4001,len(x))
    monkeypatch.setattr(research,'CACHE',tmp_path)
    monkeypatch.setattr(research,'pixel_features',lambda im,disk:np.zeros((16,16,2)))
    monkeypatch.setattr(joblib,'load',lambda path:Model())
    cold=research.probabilities(Data(),'synthetic','rf')
    warm=research.probabilities(Data(),'synthetic','rf')
    np.testing.assert_array_equal(cold,warm)
    np.testing.assert_array_equal(cold>=.4,warm>=.4)

def test_accelerated_grid_exactly_matches_unshared_geometry():
    from research import encoded_grid,encode_labels
    rng=np.random.default_rng(7);p=rng.random((64,64),dtype=np.float32)
    disk=np.zeros(p.shape,bool);disk[5:59,5:59]=True
    params=[dict(threshold=t,min_area=a,close=c) for t,a,c in itertools.product([.25,.4,.55,.7],[20,60,120],[0,1])]
    for settings,encoded in zip(params,encoded_grid(p,disk,params)):
        assert encoded==encode_labels(instances(p,disk,**settings))

def test_component_training_dates_exclude_assessment_and_validation():
    main=json.loads((HERE/'protocol.json').read_text())
    follow=json.loads((HERE/'component-protocol.json').read_text())
    groups=[{n[:8] for n in follow[k]} for k in ['train','tune']]
    groups.append({n[:8] for n in main['split']['assessment']})
    for a,b in itertools.combinations(groups,2):assert not a&b

def test_component_filter_preserves_iou_column_alignment():
    from component_followup import scores
    class Model:
        def predict(self,x):return np.array([.8,.1,.9])
    matrix=np.array([[.8,0,0],[0,0,.9]],np.float32)
    result=scores([('synthetic',np.ones((3,14)),[(1,matrix)])],Model(),.5)[0]
    assert (result['tp'],result['fp'],result['fn'])==(2,0,0)
    assert result['iou_sum']==pytest.approx(1.7)

def test_image_only_inference_needs_no_annotation_loader(monkeypatch):
    import research
    from inference import segment
    def forbidden(*args,**kwargs):raise AssertionError('Inference read annotations')
    monkeypatch.setattr(research.Dataset,'__init__',forbidden)
    labels=segment(np.zeros((2048,2048),np.uint8),'rf_filter')
    assert labels.shape==(2048,2048) and not labels.any()
