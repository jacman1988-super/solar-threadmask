import numpy as np
import inference as inf

def test_feature_schema_is_finite_and_uses_both_probability_maps():
    low=np.zeros((1024,1024),np.int32);low[400:402,300:340]=1
    labels=np.repeat(np.repeat(low,2,0),2,1);rf=np.ones(low.shape,np.float32)*.8;unet=np.ones_like(rf)*.9
    x=inf.p.features(labels,rf,unet,np.ones_like(rf)*100,np.ones_like(low,bool))
    assert x.shape==(1,22) and np.isfinite(x).all()
    assert np.isclose(x[0,14],.9) and x[0,19]==1 and x[0,20]==1

def test_learned_filter_retains_baseline_and_rejects_low_quality_addition():
    unet=np.zeros((1024,1024),np.float32);unet[400:402,300:340]=.9
    rf=np.ones_like(unet)*.8;disk=np.ones(unet.shape,bool);im=np.ones_like(unet)*100
    original=np.zeros((2048,2048),np.int32);original[1200:1210,800:850]=1
    class Filter:
        def predict(self,x):return np.ones(len(x))*.1
    result=inf.learned(original,im,disk,rf,unet,Filter(),.7,.2)
    assert np.array_equal(result,original)

def test_rejected_baseline_instance_cannot_be_reintroduced_by_supplement():
    unet=np.zeros((1024,1024),np.float32);unet[500:502,300:340]=.9
    rf=np.ones_like(unet)*.8;disk=np.ones(unet.shape,bool)
    original=np.zeros((2048,2048),np.int32);original[1200:1210,800:850]=1;original[1000:1004,600:680]=2
    result=inf.joint(original,np.array([.9,.201]),rf,unet,disk,.205,.6)
    assert np.all(result[original==1]==1) and not result[original==2].any()
