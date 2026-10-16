import numpy as np
import pytest
from replace_masks import choose
from inference import substitute

def test_ambiguous_overlap_cannot_replace_multiple_instances():
    pair=np.array([[.5,.3,0],[0,.2,.4]])
    result=choose(pair,np.array([.1,.1]),np.array([.7,.99,.8]),.1,.4,.25)
    assert result.tolist()==[0,2]

def test_substitution_preserves_instance_count_and_unselected_pixels():
    original=np.zeros((10,10),np.int32);original[1:3,1:3]=1;original[6:8,6:8]=2
    proposed=np.zeros_like(original);proposed[1:4,1:4]=1
    result=substitute(original,proposed,np.array([0,-1]))
    assert set(np.unique(result))=={0,1,2}
    assert np.all(result[original==2]==2)

def test_replacement_cannot_overwrite_unselected_instance():
    original=np.zeros((10,10),np.int32);original[1:3,1:3]=1;original[6:8,6:8]=2
    proposed=np.ones_like(original)
    with pytest.raises(ValueError,match='overlapping'):substitute(original,proposed,np.array([0,-1]))
