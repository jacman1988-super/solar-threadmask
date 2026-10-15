import unittest
import numpy as np
from solar_v2 import ConfigV2,segment_v2
from solar_v3 import candidates,filter_labels,fit_filter,probabilities,FEATURE_NAMES

class FilterTests(unittest.TestCase):
    def test_zero_threshold_preserves_exact_v2_instances(self):
        image=np.zeros((128,128),dtype=np.uint8);yy,xx=np.ogrid[:128,:128]
        image[(yy-64)**2+(xx-64)**2<54**2]=180
        image[60:64,40:85]=60;image[35:42,45:50]=70
        config=ConfigV2(resolution=128,sigma=5,low=.09,min_area=5,limb_margin=4)
        labels,x=candidates(image,config)
        self.assertGreater(len(x),0)
        np.testing.assert_array_equal(labels,segment_v2(image,config))
        np.testing.assert_array_equal(filter_labels(labels,np.zeros(len(x)),0),labels)

    def test_rejection_preserves_survivor_pixels_and_relabels(self):
        labels=np.array([[0,1,1],[2,2,3]],dtype=np.int32)
        result=filter_labels(labels,[.1,.9,.8],.5)
        np.testing.assert_array_equal(result,np.array([[0,0,0],[1,1,2]]))

    def test_empty_image_has_no_feature_rows(self):
        labels,x=candidates(np.zeros((64,64),dtype=np.uint8),ConfigV2(resolution=64,limb_margin=2))
        self.assertEqual(x.shape,(0,len(FEATURE_NAMES)))
        self.assertEqual(labels.max(),0)

    def test_fitted_probability_separates_known_classes(self):
        x=np.zeros((40,len(FEATURE_NAMES)));x[:,0]=np.linspace(-2,2,40)
        y=(x[:,0]>0).astype(int);model=fit_filter(x,y)
        score=probabilities(x,model)
        self.assertLess(score[0],.1);self.assertGreater(score[-1],.9)
        self.assertTrue(np.all(np.diff(score)>0))

    def test_mismatched_features_refused(self):
        with self.assertRaises(ValueError):fit_filter(np.zeros((2,2)),[0,1])
        with self.assertRaises(ValueError):filter_labels(np.array([[1]]),[],.5)
