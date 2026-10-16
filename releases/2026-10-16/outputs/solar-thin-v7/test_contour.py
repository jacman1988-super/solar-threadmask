import numpy as np
from shrink import shrink
from contour import refine

def test_erosion_can_recover_overwide_thin_mask_without_new_instances():
    labels=np.zeros((100,100),np.int32);labels[40:50,10:90]=1
    truth=np.zeros_like(labels,bool);truth[43:47,10:90]=1
    before=(truth&(labels>0)).sum()/(truth|(labels>0)).sum()
    result=shrink(labels,32,3);after=(truth&(result>0)).sum()/(truth|(result>0)).sum()
    assert before<.5 and after>.5
    assert set(np.unique(result))=={0,1} and not np.any((result>0)&(labels==0))

def test_erosion_cannot_delete_a_very_thin_instance():
    labels=np.zeros((100,100),np.int32);labels[40:42,10:90]=1
    assert np.array_equal(shrink(labels,8,3),labels)

def test_growth_preserves_existing_pixels_and_instance_ids():
    labels=np.zeros((100,100),np.int32);labels[40:42,10:90]=1;labels[45:47,10:90]=2
    result=refine(labels,np.ones(labels.shape),8,3,0)
    assert np.array_equal(result[labels>0],labels[labels>0])
    assert set(np.unique(result))=={0,1,2}
