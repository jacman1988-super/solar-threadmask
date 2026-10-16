"""CPU baseline: local dark contrast and seeded component selection, no training."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class Config:
    background_sigma: float = 16.0
    low_z: float = 2.0
    high_z: float = 4.0
    min_area: int = 24
    disk_fraction: float = 0.95


def segment(image, config=Config()):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Expected finite 2D grayscale image')
    if config.background_sigma <= 0 or not 0 < config.low_z <= config.high_z or config.min_area < 1:
        raise ValueError('Invalid contrast/area configuration')
    if not 0 < config.disk_fraction <= 1:
        raise ValueError('Invalid disk fraction')
    y, x = np.indices(image.shape)
    h, w = image.shape
    disk = ((x-(w-1)/2)**2+(y-(h-1)/2)**2) <= (min(h,w)*config.disk_fraction/2)**2
    background = ndi.gaussian_filter(image, config.background_sigma)
    residual = background - image
    center = np.median(residual[disk])
    scale = max(float(np.median(np.abs(residual[disk]-center))) * 1.4826, 1.0)
    contrast = (residual-center)/scale
    labels, count = ndi.label((contrast >= config.low_z) & disk, structure=np.ones((3,3)))
    areas = np.bincount(labels.ravel())
    seeds = set(np.unique(labels[(contrast >= config.high_z) & disk])) - {0}
    # No closing or dilation: avoid deliberately connecting neighboring filaments.
    keep = [i for i in range(1,count+1) if i in seeds and areas[i] >= config.min_area]
    mapping = np.zeros(count+1, dtype=np.int32)
    mapping[keep] = np.arange(1,len(keep)+1)
    return mapping[labels]


def masks(labels):
    return [labels == i for i in np.unique(labels) if i != 0]


def pq(truth, predicted):
    """Local diagnostic PQ, IoU strictly > .5; not organizer-equivalence certified."""
    truth, predicted = list(truth), list(predicted)
    if not truth and not predicted:
        return {'pq': None, 'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0}
    shape = (truth or predicted)[0].shape
    if any(m.shape != shape or m.dtype != np.bool_ for m in truth + predicted):
        raise ValueError('All masks must have the same shape and bool dtype')
    pairs = []
    for i, gt in enumerate(truth):
        for j, pr in enumerate(predicted):
            union = np.count_nonzero(gt | pr)
            iou = np.count_nonzero(gt & pr)/union if union else 0
            if iou > 0.5:
                pairs.append((iou,i,j))
    return match_pq(len(truth),len(predicted),pairs)


def match_pq(n_truth, n_predicted, pairs):
    if not n_truth and not n_predicted:
        return {'pq': None, 'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0}
    # Predicted components are disjoint. COCO annotation overlap can differ; exact
    # matching maximizes cardinality first and then IoU for these local diagnostics.
    from scipy.optimize import linear_sum_assignment
    weights = np.zeros((n_truth, n_predicted))
    for iou,i,j in pairs:
        weights[i,j] = 1 + iou / (max(n_truth,n_predicted)+1)
    if weights.size:
        rows, cols = linear_sum_assignment(weights, maximize=True)
        matched = [(i,j) for i,j in zip(rows,cols) if weights[i,j] > 0]
    else:
        matched = []
    tp = len(matched)
    scores = {(i,j): iou for iou,i,j in pairs}
    total = sum(scores[i,j] for i,j in matched)
    fp, fn = n_predicted-tp, n_truth-tp
    return {'pq': total/(tp+0.5*fp+0.5*fn), 'tp':tp, 'fp':fp, 'fn':fn, 'iou_sum':total}
