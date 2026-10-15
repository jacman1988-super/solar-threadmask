import unittest
import numpy as np
from solar_precision_guard import guard_mask
from solar_v3 import FEATURE_NAMES,filter_labels

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.model={'feature_names':FEATURE_NAMES,'mean':[0]*11,'scale':[1]*11,'weights':[0]*12}
        self.rule={'feature':8,'name':'mean_disk_gap','direction':'ge','cutoff':.02}
    def test_inclusive_boundary_and_contiguous_relabel(self):
        x=np.zeros((3,11));x[:,8]=[.019,.02,.03]
        keep=guard_mask(x,self.model,self.rule)
        self.assertEqual(keep.tolist(),[False,True,True])
        self.assertEqual(filter_labels(np.array([[0,1,2,3]]),keep.astype(float),.5).tolist(),[[0,0,1,2]])
    def test_probability_filter_still_applies(self):
        model=dict(self.model,weights=[-10]+[0]*11)
        self.assertFalse(guard_mask(np.ones((2,11)),model,self.rule).any())
    def test_schema_mismatch_rejected(self):
        with self.assertRaises(ValueError):guard_mask(np.zeros((1,11)),self.model,dict(self.rule,name='wrong'))
