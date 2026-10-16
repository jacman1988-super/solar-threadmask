import unittest
import types
import numpy as np
from organizer_metric import stats_from_iou,pool,instance_stats
from reference.organizer_pq import get_pq_score
from pycocotools import mask as mask_api


class TensorAdapter(np.ndarray):
    """Only sum(dim=...) compatibility for unchanged pure reference functions."""
    def sum(self,*args,dim=None,**kwargs):
        if dim is not None:kwargs['axis']=dim
        return super().sum(*args,**kwargs)


class Table:
    def __init__(self,matrices):self.matrices=matrices
    def itertuples(self,index=False):
        for m in self.matrices:
            yield types.SimpleNamespace(iou_matrix=np.asarray(m,dtype=np.float32).view(TensorAdapter),n_gt=m.shape[0],n_pred=m.shape[1])


class MetricTests(unittest.TestCase):
    def compare(self,matrices):
        self.assertAlmostEqual(get_pq_score(Table(matrices)),pool(stats_from_iou(m) for m in matrices)['pq'],places=12)

    def test_reference_random_matrices(self):
        rng=np.random.default_rng(42)
        for _ in range(100):
            self.compare([rng.random((int(rng.integers(0,8)),int(rng.integers(0,8))),dtype=np.float32) for _ in range(4)])

    def test_empty_and_threshold(self):
        self.compare([np.zeros((0,0)),np.zeros((0,2)),np.zeros((3,0)),np.array([[.5,.50001]])])

    def test_multiple_hits_counted(self):
        stats=stats_from_iou(np.array([[.6],[.7]]))
        self.assertEqual(stats['tp'],2)
        self.compare([np.array([[.6],[.7]])])

    def test_pooled_is_not_image_average(self):
        first=stats_from_iou(np.ones((1,1)));second=stats_from_iou(np.zeros((10,10)))
        self.assertAlmostEqual(pool([first,second])['pq'],1/11)

    def test_rle_overlap_equals_dense_float32(self):
        rng=np.random.default_rng(6)
        gt=[rng.random((16,20))>.7 for _ in range(3)]
        pr=[gt[0],rng.random((16,20))>.7]
        enc=lambda x:mask_api.encode(np.asfortranarray(x.astype(np.uint8)))
        g=np.stack(gt).astype(np.float32).reshape(3,-1);p=np.stack(pr).astype(np.float32).reshape(2,-1)
        inter=g@p.T;dense=inter/(g.sum(axis=1)[:,None]+p.sum(axis=1)[None,:]-inter)
        self.assertEqual(instance_stats(list(map(enc,gt)),list(map(enc,pr))),stats_from_iou(dense))
