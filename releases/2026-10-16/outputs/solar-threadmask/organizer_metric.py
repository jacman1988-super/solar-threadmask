"""CPU implementation of public self-evaluation v6 pooling (not private scorer)."""
import numpy as np
from pycocotools import mask as coco_mask


def stats_from_iou(iou):
    iou=np.asarray(iou,dtype=np.float32)
    if iou.ndim!=2:raise ValueError('Expected GT x prediction matrix')
    hit=iou>0.5
    n_gt,n_pred=iou.shape
    # Unlike Hungarian matching, public v6 counts every qualifying pair.
    return {'tp':int(hit.sum()),'fp':int(np.count_nonzero(hit.sum(axis=0)==0)),
            'fn':int(np.count_nonzero(hit.sum(axis=1)==0)),
            'iou_sum':float(sum(iou[hit].tolist())),
            'gt_split_gt0':int(np.count_nonzero((iou>0).sum(axis=1)>1)),
            'pred_merge_gt0':int(np.count_nonzero((iou>0).sum(axis=0)>1)),
            'gt_count':n_gt,'pred_count':n_pred}


def instance_stats(truth,predicted):
    iou=(np.asarray(coco_mask.iou(truth,predicted,[0]*len(predicted)),dtype=np.float32)
         if truth and predicted else np.zeros((len(truth),len(predicted)),dtype=np.float32))
    return stats_from_iou(iou)


def pool(stats):
    stats=list(stats)
    tp=sum(r['tp'] for r in stats);fp=sum(r['fp'] for r in stats);fn=sum(r['fn'] for r in stats)
    iou_sum=sum(r['iou_sum'] for r in stats);den=tp+0.5*fp+0.5*fn
    return {'pq':iou_sum/den if den else 0.0,'tp':tp,'fp':fp,'fn':fn,'iou_sum':iou_sum,
            'gt_split_gt0':sum(r.get('gt_split_gt0',0) for r in stats),
            'pred_merge_gt0':sum(r.get('pred_merge_gt0',0) for r in stats)}
