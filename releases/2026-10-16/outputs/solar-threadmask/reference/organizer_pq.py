# Extracted from Azim Ahmadzadeh's Self_Evaluation_Notebook, version 6.
# https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
# Notebook displayed Apache 2.0 license. Only these three pure scoring functions
# are retained. Function bodies are unchanged; annotations are deferred so tests
# can use a NumPy tensor adapter without installing torch.
from __future__ import annotations

def fp_count_hit(hit_matrix: torch.Tensor) -> int:
    """
    Counts false-positive predictions for one image from a matching matrix where
    the value is 1 if the predicted filament overlaps with the GT filament and 0 otherwise.

    A predicted filament is a false positive if it does not hit any GT filament:
    the sum of its column is zero.

    Example (rows = GT, columns = predictions):

        hit_matrix = [[0, 0, 1],
                      [0, 0, 0],
                      [0, 0, 1]]

        # column sums: [0, 0, 2]  ->  columns 0 and 1 are FPs
        fp_count_hit(hit_matrix) -> 2

    Args:
        hit_matrix: Boolean tensor of shape (n_gt, n_pred), match indicators per pair.

    Returns:
        Scalar integer tensor: number of predictions with no qualifying GT overlap.
    """
    gt_matches_per_pred = hit_matrix.sum(dim=0)
    return (gt_matches_per_pred == 0).sum().item()

def fn_count_hit(hit_matrix: torch.Tensor) -> int:
    """
    Counts false-negative annotations for one image from a matching matrix where
    the value is 1 if the predicted filament overlaps with the GT filament and 0 otherwise.

    A GT filament is a false negative if it does not hit any predicted filament: the sum of
    its row is zero.

    Example (rows = GT, columns = predictions):

        hit_matrix = [[0, 0, 1],
                      [0, 0, 0],
                      [0, 0, 1]]

        # row sums: [1, 0, 1]  ->  row 1 is FN
        fn_count_hit(hit_matrix) -> 1

    Args:
        hit_matrix: Boolean tensor of shape (n_gt, n_pred), match indicators per pair.

    Returns:
        Scalar integer tensor: number of annotations with no qualifying prediction overlap.
    """
    pred_matches_per_gt = hit_matrix.sum(dim=1)
    return (pred_matches_per_gt == 0).sum().item()

def get_pq_score(overlap_df: pd.DataFrame) -> float:
    """
    Compute the Panoptic Quality (PQ) score over all images.

    A GT/prediction pair is a true positive (TP) when its IoU exceeds the
    IoU threshold (0.5). Predictions with no match are false positives (FP),
    GT filaments with no match are false negatives (FN), and:

        PQ = sum(IoU of TP pairs) / (|TP| + 0.5 * |FP| + 0.5 * |FN|)

    Args:
        overlap_df: Per-image overlap table as produced by ``get_overlap_df``;
            must contain columns ``iou_matrix`` (IoU tensor of shape
            (n_gt, n_pred)), ``n_gt`` (GT filament count), and ``n_pred``
            (predicted filament count).

    Returns:
        Panoptic Quality score in ``[0, 1]`` (higher is better), or ``0.0`` if
        the denominator is zero.
    """
    iou_threshold: float = 0.5

    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0

    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred

        if n_gt == 0:
            fp_count += n_pred
            continue

        if n_pred == 0:
            fn_count += n_gt
            continue

        hit_matrix = iou_matrix > iou_threshold

        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())

        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count

    if denominator > 0:
        pq_score = sum(tp_iou_scores) / denominator
    else:
        pq_score = 0.0

    return pq_score
