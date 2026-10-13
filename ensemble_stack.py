"""Step 1 build-up on the current best (fuse feat + RAC = 0.921):
  1a  add mammoscreen's own SCORE as a 3rd component (grid-weighted, tuned on train)
  1b  STACKING meta-learner (logistic on the component scores) + percentile calibration
Each reported in isolation vs the feat+RAC reference. Honest: weights/threshold/meta
all fit on TRAIN (feat/RAC use out-of-fold scores); evaluated once on test paired-151.
"""
import itertools
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config
import multiview_probe as MV
from mil import _best_threshold, _metrics

BASE = ["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"]


def _pct(x, ref):
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


def get():
    MV.run(["mammoscreen_feat"], dataset="cbis_full", mode="fourway", C=0.2)
    fm_te, fm_tr = MV.run.last_cf, MV.run.last_train_cf
    MV.run(BASE, dataset="cbis_full", mode="fourway", C=0.5)
    rc_te, rc_tr = MV.run.last_cf, MV.run.last_train_cf
    d_tr = np.load(os.path.join(config.PROJECT_DIR, "cache", "mammoscreen_train.npz"), allow_pickle=True)
    d_te = np.load(os.path.join(config.PROJECT_DIR, "cache", "mammoscreen_test.npz"), allow_pickle=True)
    ms_tr = {str(c): float(s) for c, s in zip(d_tr["cids"], d_tr["s"])}
    ms_te = {str(c): float(s) for c, s in zip(d_te["cids"], d_te["s"])}

    def build(cf_f, cf_r, msmap):
        cids = np.array(cf_f["cids"]); y = np.array(cf_f["y_true"]); pidx = np.array(cf_f["paired_idx"])
        keep = np.array([i for i in pidx if cids[i] in msmap])
        return (y[keep], np.array(cf_f["y_score"])[keep], np.array(cf_r["y_score"])[keep],
                np.array([msmap[cids[i]] for i in keep]))

    return build(fm_tr, rc_tr, ms_tr), build(fm_te, rc_te, ms_te)


def report(name, ytr, ftr, yte, fte):
    thr = _best_threshold(ytr, ftr)
    m = _metrics(yte, (fte >= thr).astype(int), fte)
    ob = (0, 0)
    for t in np.unique(fte):
        mm = _metrics(yte, (fte >= t).astype(int), fte)
        if mm["bal_acc"] > ob[0]:
            ob = (mm["bal_acc"], mm["accuracy"])
    print(f"  {name:36s} AUC={roc_auc_score(yte, fte):.4f} | honest acc={m['accuracy']:.4f} "
          f"bal={m['bal_acc']:.4f} | best-thr acc={ob[1]:.4f} bal={ob[0]:.4f}")


def main():
    (ytr, Ftr, Rtr, Mtr), (yte, Fte, Rte, Mte) = get()
    Fp, Rp, Mp = _pct(Ftr, Ftr), _pct(Rtr, Rtr), _pct(Mtr, Mtr)
    Fpe, Rpe, Mpe = _pct(Fte, Ftr), _pct(Rte, Rtr), _pct(Mte, Mtr)

    print("=== Step 1 build-up on feat+RAC (0.921) ===")
    report("feat+RAC (ref, w=0.8)", ytr, 0.8 * Fp + 0.2 * Rp, yte, 0.8 * Fpe + 0.2 * Rpe)

    # 1a: add mammoscreen score, grid weights tuned on train AUC
    best = None
    for w in itertools.product([0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0], repeat=3):
        if sum(w) == 0:
            continue
        wn = np.array(w) / sum(w)
        a = roc_auc_score(ytr, wn[0] * Fp + wn[1] * Rp + wn[2] * Mp)
        if best is None or a > best[0]:
            best = (a, wn)
    w = best[1]
    report(f"feat+RAC+score (w={np.round(w, 2)})", ytr, w[0]*Fp+w[1]*Rp+w[2]*Mp,
           yte, w[0]*Fpe+w[1]*Rpe+w[2]*Mpe)

    # 1b: stacking meta-learner (logistic) on the 3 component scores
    Xtr, Xte = np.stack([Ftr, Rtr, Mtr], 1), np.stack([Fte, Rte, Mte], 1)
    s = StandardScaler().fit(Xtr)
    clf = LogisticRegression(class_weight="balanced", max_iter=3000).fit(s.transform(Xtr), ytr)
    report("STACK logistic(feat,RAC,score)", ytr, clf.predict_proba(s.transform(Xtr))[:, 1],
           yte, clf.predict_proba(s.transform(Xte))[:, 1])

    Xtrp, Xtep = np.stack([Fp, Rp, Mp], 1), np.stack([Fpe, Rpe, Mpe], 1)
    clf2 = LogisticRegression(class_weight="balanced", max_iter=3000).fit(Xtrp, ytr)
    report("STACK logistic(pct feat,RAC,score)", ytr, clf2.predict_proba(Xtrp)[:, 1],
           yte, clf2.predict_proba(Xtep)[:, 1])


if __name__ == "__main__":
    main()
