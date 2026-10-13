"""Stage 4b: fuse retrieved neighbour labels into a binary prediction.

Three training-free heads (no parameters learned):
  vote     - plain majority over k neighbour labels
  weighted - softmax over similarity -> weighted vote
  tip      - Tip-Adapter affinity A = exp(-beta*(1 - cosine)); with a frozen
             encoder and no classifier prior this is an exp-weighted vote.

All heads return (pred[N], score_pos[N]) where score_pos is the weight-mass on
the malignant class (1) — used both for thresholding and for AUC.
"""
import numpy as np

import config


def _to_affinity(scores, distance, head, beta=config.TIP_BETA):
    """Map raw search scores to non-negative per-neighbour weights."""
    if head == "vote":
        return np.ones_like(scores, dtype=np.float64)
    if distance == "cosine":
        cos = scores  # similarity in [-1,1]
        if head == "tip":
            return np.exp(-beta * (1.0 - cos))
        z = cos - cos.max(axis=1, keepdims=True)   # weighted: softmax over sim
        return np.exp(z)
    else:  # l2: scores are squared L2 (smaller = closer)
        d = scores
        if head == "tip":
            return np.exp(-beta * d / (d.max() + 1e-8))
        z = -d + d.min(axis=1, keepdims=True)
        return np.exp(z)


def fuse(scores, nbr_labels, distance, head):
    """scores, nbr_labels: [N,k]. Returns (pred[N], score_pos[N])."""
    w = _to_affinity(scores, distance, head)                # [N,k] >= 0
    w_sum = np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    pos_mass = (w * (nbr_labels == 1)).sum(axis=1, keepdims=True) / w_sum
    score_pos = pos_mass.ravel()
    pred = (score_pos >= 0.5).astype(np.int64)
    return pred, score_pos
