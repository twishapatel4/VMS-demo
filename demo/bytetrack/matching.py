import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))

    # scipy returns the optimal assignment (no cost threshold)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_a, unmatched_b = [], [], []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            matched_rows.add(r)
            matched_cols.add(c)
    for r in range(cost_matrix.shape[0]):
        if r not in matched_rows:
            unmatched_a.append(r)
    for c in range(cost_matrix.shape[1]):
        if c not in matched_cols:
            unmatched_b.append(c)
    matches = np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return matches, tuple(unmatched_a), tuple(unmatched_b)


def ious(atlbrs, btlbrs):
    """Computes IoU matrix between two sets of [x1, y1, x2, y2] boxes."""
    ious_mat = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious_mat.size == 0:
        return ious_mat
    a = np.asarray(atlbrs, dtype=np.float32)
    b = np.asarray(btlbrs, dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    for i in range(len(a)):
        xx1 = np.maximum(a[i, 0], b[:, 0])
        yy1 = np.maximum(a[i, 1], b[:, 1])
        xx2 = np.minimum(a[i, 2], b[:, 2])
        yy2 = np.minimum(a[i, 3], b[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = area_a[i] + area_b - inter
        ious_mat[i] = inter / np.maximum(union, 1e-6)
    return ious_mat


def iou_distance(atracks, btracks):
    """1 - IoU as cost. atracks/btracks may be STracks or raw tlbr lists."""
    if len(atracks) > 0 and isinstance(atracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [t.tlbr for t in atracks]
        btlbrs = [t.tlbr for t in btracks]
    _ious = ious(atlbrs, btlbrs)
    return 1 - _ious


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([d.score for d in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim
