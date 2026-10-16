import numpy as np
from consensus_runtime import combine

def test_forest_confirmation_removes_weak_proposal_without_touching_baseline():
    p=np.zeros((1024,1024),np.float32);rf=np.zeros_like(p);disk=np.ones_like(p,bool)
    p[400:402,300:340]=.8;p[500:502,300:340]=.8
    rf[400:402,300:340]=.8;rf[500:502,300:340]=.3
    base=np.zeros((2048,2048),np.int32);base[1200:1210,800:850]=1
    result=combine(base,p,rf,disk,.6)
    assert np.array_equal(result[base>0],base[base>0])
    assert result[800,600]==2 and result[1000,600]==0
    assert result.max()==2

def test_stricter_support_cannot_add_more_pixels():
    p=np.zeros((1024,1024),np.float32);p[400:402,300:340]=.8
    rf=np.ones_like(p)*.5;disk=np.ones_like(p,bool);base=np.zeros((2048,2048),np.int32)
    assert combine(base,p,rf,disk,.4).any()
    assert not combine(base,p,rf,disk,.6).any()
