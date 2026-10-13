"""Honest fused result: baseline 4-way multi-view head + mammoscreen zero-shot,
late-fused. Fusion weight and threshold are chosen on TRAIN only (percentiles are
mapped through the TRAIN score distribution — non-transductive), then applied once
to the official CBIS test set. Reports test accuracy + AUC (paired-151).
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import config
import multiview_probe as MV
from mil import _best_threshold, _metrics

BASE = ["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"]
CACHE = os.path.join(config.PROJECT_DIR, "cache")


def _pct_ref(x, ref):
    """Percentile of each x relative to the TRAIN (ref) distribution — [0,1], non-transductive."""
    ref_sorted = np.sort(ref)
    return np.searchsorted(ref_sorted, x, side="right") / max(len(ref), 1)


def _align(cf, mmap):
    cids = np.array(cf["cids"]); y = np.array(cf["y_true"])
    sb = np.array(cf["y_score"]); pidx = np.array(cf["paired_idx"])
    keep = np.array([i for i in pidx if cids[i] in mmap])
    return y[keep], sb[keep], np.array([mmap[cids[i]] for i in keep])


def main(C=0.2):
    MV.run(BASE, dataset="cbis_full", mode="fourway", C=C)
    d_tr = np.load(os.path.join(CACHE, "mammoscreen_train.npz"))
    d_te = np.load(os.path.join(CACHE, "mammoscreen_test.npz"))
    m_tr = {str(c): float(s) for c, s in zip(d_tr["cids"], d_tr["s"])}
    m_te = {str(c): float(s) for c, s in zip(d_te["cids"], d_te["s"])}
    ytr, btr, mtr = _align(MV.run.last_train_cf, m_tr)
    yte, bte, mte = _align(MV.run.last_cf, m_te)

    print(f"aligned paired cases: train {len(ytr)}, test {len(yte)}")
    print(f"  baseline head       : train-AUC {roc_auc_score(ytr, btr):.4f}  test-AUC {roc_auc_score(yte, bte):.4f}")
    print(f"  mammoscreen zero-shot: train-AUC {roc_auc_score(ytr, mtr):.4f}  test-AUC {roc_auc_score(yte, mte):.4f}")

    b_tr_p, b_te_p = _pct_ref(btr, btr), _pct_ref(bte, btr)
    m_tr_p, m_te_p = _pct_ref(mtr, mtr), _pct_ref(mte, mtr)
    best = None
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        auc_tr = roc_auc_score(ytr, w * b_tr_p + (1 - w) * m_tr_p)   # pick w on TRAIN
        if best is None or auc_tr > best[0]:
            best = (auc_tr, w)
    w = best[1]
    ftr, fte = w * b_tr_p + (1 - w) * m_tr_p, w * b_te_p + (1 - w) * m_te_p
    thr = _best_threshold(ytr, ftr)
    m = _metrics(yte, fte, thr)
    print(f"  FUSED (w_base={w} chosen by train-AUC, thr={thr:.3f}):")
    print(f"    TEST paired({len(yte)}): acc={m['accuracy']:.4f} bal_acc={m['bal_acc']:.4f} "
          f"auc={roc_auc_score(yte, fte):.4f} sens={m['sensitivity']:.4f} spec={m['specificity']:.4f}")


if __name__ == "__main__":
    main()
