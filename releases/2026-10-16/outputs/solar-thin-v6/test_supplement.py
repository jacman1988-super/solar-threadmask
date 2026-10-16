import numpy as np
from supplement import proposals

def test_disjoint_thin_proposal_and_rejection_near_baseline():
    p=np.zeros((1024,1024),np.float32);disk=np.ones_like(p,bool)
    p[400:402,300:340]=.8;p[500:502,300:340]=.8
    base=np.zeros((2048,2048),np.int32);base[998:1002,600:680]=1
    labels,items=proposals(p,disk,base,.7)
    assert len(items)==1
    k,elong,width=items[0]
    assert elong>3 and width<4
    assert labels[800,600]==k

def test_short_noise_is_not_a_thin_proposal():
    p=np.zeros((1024,1024),np.float32);p[400:404,400:404]=.9
    _,items=proposals(p,np.ones_like(p,bool),np.zeros((2048,2048),np.int32),.7)
    assert not items
