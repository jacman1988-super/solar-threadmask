import unittest
import csv
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from threadmask import Config, masks, pq, segment
from pipeline import holdout, decode, evaluate, predict, pq_encoded, encode_labels


class SolarTests(unittest.TestCase):
    def test_blank_is_empty(self):
        self.assertFalse(segment(np.full((128,128),180)).any())

    def test_two_separate_threads(self):
        im = np.full((128,128),180,dtype=float)
        im[30:90,48:51] = 100
        im[30:90,76:79] = 100
        out = masks(segment(im,Config(background_sigma=8,min_area=10)))
        self.assertEqual(len(out),2)
        self.assertTrue(all(m.sum() >= 150 for m in out))

    def test_invalid_rgb(self):
        with self.assertRaises(ValueError): segment(np.zeros((2,2,3)))

    def test_invalid_nan(self):
        with self.assertRaises(ValueError): segment(np.full((2,2),np.nan))

    def test_pq_identity_and_miss(self):
        mask = np.eye(5,dtype=bool)
        self.assertEqual(pq([mask],[mask])['pq'],1)
        self.assertEqual(pq([mask],[])['pq'],0)
        self.assertEqual(pq([],[mask])['fp'],1)
        self.assertIsNone(pq([],[])['pq'])

    def test_fragmentation_penalty(self):
        gt = np.ones((4,4),dtype=bool)
        a = gt.copy(); a[3] = False
        b = ~a
        out = pq([gt],[a,b])
        self.assertEqual(out['tp'],1)
        self.assertEqual(out['fp'],1)
        self.assertAlmostEqual(out['pq'],0.5)

    def test_compressed_rle_roundtrip(self):
        mask = np.zeros((17,23),dtype=np.uint8); mask[2:8,3:12] = 1
        rle = mask_api.encode(np.asfortranarray(mask))
        self.assertTrue(np.array_equal(decode({'segmentation':rle},17,23),mask))

    def test_date_split_groups_instruments(self):
        self.assertEqual(holdout('20260101120000Bh.jpeg'),holdout('20260101235959Mh.jpeg'))

    def test_encoded_metric_matches_dense(self):
        gt = np.ones((4,4),dtype=bool)
        a = gt.copy(); a[3] = False
        b = ~a
        encode = lambda m: mask_api.encode(np.asfortranarray(m.astype(np.uint8)))
        self.assertEqual(pq([gt],[a,b]),pq_encoded([encode(gt)],[encode(a),encode(b)]))

    def test_sparse_encoding_exact_pixels(self):
        rng=np.random.default_rng(42)
        for labels in [np.zeros((17,23),dtype=np.int32),np.ones((17,23),dtype=np.int32),
                       rng.integers(0,5,size=(17,23),dtype=np.int32)]:
            encoded=encode_labels(labels)
            expected=[labels==i for i in np.unique(labels) if i]
            self.assertEqual(len(encoded),len(expected))
            for rle,mask in zip(encoded,expected):
                self.assertTrue(np.array_equal(mask_api.decode(rle).astype(bool),mask))

    def test_synthetic_end_to_end(self):
        # Original artificial image, explicitly not solar-observation evidence.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'train/train_images').mkdir(parents=True)
            (root/'test/test_images').mkdir(parents=True)
            day = next(f'202601{d:02d}' for d in range(1,29) if holdout(f'202601{d:02d}120000Bh.jpeg'))
            name = day+'120000Bh.jpeg'
            image = np.full((2048,2048),180,dtype=np.uint8)
            image[600:900,900:906] = 70
            Image.fromarray(image).save(root/'train/train_images'/name)
            Image.fromarray(image).save(root/'test/test_images'/name)
            annotation = {'images':[{'id':'batch-'+day,'file_name':name,'height':2048,'width':2048}],
                          'annotations':[{'image_id':'batch-'+day,'segmentation':[[900,600,906,600,906,900,900,900]]}]}
            (root/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').write_text(json.dumps(annotation))
            evaluate(root,root/'validation.json')
            self.assertEqual(len(json.loads((root/'validation.json').read_text())['rows']),1)
            predict(root,root/'candidate.csv')
            with (root/'candidate.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows),0)
            self.assertEqual(len({r['filament_id'] for r in rows}),len(rows))
            for row in rows:
                decoded = mask_api.decode({'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')})
                self.assertGreater(decoded.sum(),0)


if __name__ == '__main__': unittest.main()
