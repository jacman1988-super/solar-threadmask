import numpy as np
import pytest
from ridge import features,refine

@pytest.mark.parametrize('sigma',[1,2,4])
def test_dark_thin_ridges_have_positive_response(sigma):
    x=np.arange(128)-64;line=np.exp(-x*x/(2*sigma*sigma));dark=np.broadcast_to(180-60*line,(128,128)).copy();bright=np.broadcast_to(120+60*line,(128,128)).copy();disk=np.ones(dark.shape,bool)
    a=features(dark,disk);b=features(bright,disk)
    assert a.shape==(128,128,10) and np.isfinite(a).all()
    channel={1:4,2:6,4:8}[sigma]
    assert a[64,64,channel]>0 and a[64,64,channel]>b[64,64,channel]

def test_pixel_refinement_recovers_narrow_contour_without_new_ids():
    labels=np.zeros((100,100),np.int32);labels[40:50,10:90]=1;labels[70:80,70:80]=2
    prob=np.zeros(labels.shape,np.float32);prob[43:47,10:90]=1
    result=refine(labels,prob,32,.5,0)
    assert np.array_equal(result==1,prob>0)
    assert np.array_equal(result==2,labels==2)
    assert set(np.unique(result))=={0,1,2}

def test_low_support_cannot_delete_instance():
    labels=np.zeros((100,100),np.int32);labels[40:50,10:90]=1
    assert np.array_equal(refine(labels,np.zeros(labels.shape),32,.5,4),labels)
