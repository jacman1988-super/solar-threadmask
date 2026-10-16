import unittest
import numpy as np
from pycocotools import mask as mask_api
from audit_candidate import validate_rows

def row(identifier,mask):
    rle=mask_api.encode(np.asfortranarray(np.asarray(mask,dtype=np.uint8)))
    return {'filament_id':identifier,'segmentation_rle':rle['counts'].decode('ascii')}

class CandidateAuditTests(unittest.TestCase):
    def test_zero_detection_image_is_valid_without_fake_row(self):
        result=validate_rows([row('a_1',[[1,0],[0,0]])],{'a':1,'b':0},(2,2))
        self.assertEqual(result['zero_detection_images'],1)
    def test_unknown_image_rejected_even_with_right_total(self):
        with self.assertRaises(ValueError):validate_rows([row('wrong_1',[[1,0],[0,0]])],{'a':1},(2,2))
    def test_overlap_between_different_ids_rejected(self):
        mask=[[1,0],[0,0]]
        with self.assertRaises(ValueError):validate_rows([row('a_1',mask),row('a_2',mask)],{'a':2},(2,2))
    def test_empty_instance_is_rejected(self):
        with self.assertRaises(ValueError):validate_rows([row('a_1',[[0,0],[0,0]])],{'a':1},(2,2))
    def test_extra_csv_column_is_rejected(self):
        record=row('a_1',[[1,0],[0,0]]);record[None]=['unexpected-column']
        with self.assertRaises(ValueError):validate_rows([record],{'a':1},(2,2))
    def test_missing_rle_is_rejected_before_decoder(self):
        with self.assertRaises(ValueError):validate_rows([{'filament_id':'a_1','segmentation_rle':None}],{'a':1},(2,2))
