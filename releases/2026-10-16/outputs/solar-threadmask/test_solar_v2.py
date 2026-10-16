import unittest
import numpy as np
from solar_v2 import ConfigV2,features,segment_v2


class DiskAwareTests(unittest.TestCase):
    def fixture(self):
        y,x=np.indices((128,128));disk=(x-70)**2+(y-58)**2<48**2
        return np.where(disk,180,0).astype(np.uint8),disk

    def test_offset_disk_and_limb_rejection(self):
        image,disk=self.fixture()
        score,inside=features(image,8,128,8)
        self.assertFalse(inside[~disk].any())
        self.assertTrue(inside[58,70])
        self.assertFalse(inside[58,116])
        self.assertLess(float(np.max(np.abs(score[inside]))),.001)

    def test_separate_dark_filaments_detected(self):
        image,disk=self.fixture();image[40:68,58:62]=130;image[40:68,78:82]=130
        labels=segment_v2(image,ConfigV2(resolution=128,sigma=8,limb_margin=8,min_area=10))
        self.assertEqual(int(labels.max()),2)
        self.assertFalse(labels[~disk].any())

    def test_empty_observation(self):
        self.assertFalse(segment_v2(np.zeros((128,128),dtype=np.uint8),ConfigV2(resolution=128)).any())

    def test_invalid_input(self):
        with self.assertRaises(ValueError):features(np.zeros((32,32,3)))


if __name__=='__main__':unittest.main()
