import numpy as np
from experiment import hysteresis

def test_faint_thin_continuation_is_retained_but_unseeded_noise_is_rejected():
    p=np.zeros((100,100),np.float32);disk=np.ones_like(p,bool)
    p[40:42,20:60]=.45;p[40:42,20:25]=.8
    p[70:72,20:60]=.45
    mask=hysteresis(p,disk,.4)[::2,::2]
    assert np.all(mask[40:42,20:60]==1)
    assert not mask[70:72,20:60].any()

def test_sub_area_noise_is_rejected_even_when_confident():
    p=np.zeros((100,100));p[40:42,40:43]=.9
    assert not hysteresis(p,np.ones_like(p,bool),.4).any()

def test_growth_stays_inside_probability_support_and_disk_margin():
    p=np.zeros((100,100));p[30:32,30:50]=.9
    disk=np.zeros_like(p,bool);disk[15:85,15:85]=1
    m=hysteresis(p,disk,.4)[::2,::2]
    assert m.any();assert not m[p<.4].any();assert not m[~disk].any()
