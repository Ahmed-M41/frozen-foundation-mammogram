"""Ablation: does mammoscreen (zero-shot cancer score) help when late-fused with
our 4-way multi-view head? Rank-average the two case scores on CBIS paired-151.
Reports baseline / mammoscreen / fused AUC so we see mammoscreen's isolated effect.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import multiview_probe as MV
import mammoscreen

BASE = ["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"]


def _ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def main(C=0.2):
    MV.run(BASE, dataset="cbis_full", mode="fourway", C=C)   # populates run.last_cf
    cf = MV.run.last_cf
    cids = np.array(cf["cids"]); y = np.array(cf["y_true"])
    sb = np.array(cf["y_score"]); pidx = np.array(cf["paired_idx"])

    mc_cids, _, mc_s, _ = mammoscreen.case_scores(paired_only=True)
    mmap = {c: s for c, s in zip(mc_cids, mc_s)}

    keep = np.array([i for i in pidx if cids[i] in mmap])
    yy = y[keep]
    base = sb[keep]
    mc = np.array([mmap[cids[i]] for i in keep])

    print(f"aligned paired cases: {len(keep)}")
    print(f"  baseline (our 4-way head) AUC : {roc_auc_score(yy, base):.4f}")
    print(f"  mammoscreen zero-shot     AUC : {roc_auc_score(yy, mc):.4f}")
    print("  --- rank-average late fusion ---")
    for w in [0.3, 0.5, 0.7]:
        f = w * _ranks(base) + (1 - w) * _ranks(mc)
        print(f"    fused (w_base={w}, w_mammoscreen={1 - w:.1f}): AUC={roc_auc_score(yy, f):.4f}")


if __name__ == "__main__":
    main()
