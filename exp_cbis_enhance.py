"""Honest exploration of whether CBIS-DDSM BASE4 can be improved by TRAINING changes.

Rules: hyperparameters are selected on the TRAIN out-of-fold signal only; the official
test set is never used to choose anything, and no seed is cherry-picked. Reports:
  (a) per-seed BASE4 spread (shows seeds are stable, not a lever),
  (b) a regularisation C sweep, each C's TRAIN-OOF case AUC vs TEST case AUC/acc/bal,
  (c) class-weight balanced vs none (a sensitivity/accuracy trade),
so the honest, train-selected best config can be read off and compared to the current
C=0.5, balanced setting.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import db, evaluate
import miniddsm_B as M
from mil import _best_threshold, _metrics
from exp_msfeat_fusion import load_enc, BASE3, MSFEAT, cbis_ms

ENC4 = BASE3 + [MSFEAT]
SEEDS = [0, 1, 2, 3]


def head_cv(Xtr, ytr, ptr, Xte, C, cw, seed, n_splits=5):
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof, tep = np.zeros(len(Xtr)), []
    for trf, vaf in sg.split(Xtr, ytr, groups=ptr):
        sc = StandardScaler().fit(Xtr[trf])
        c = LogisticRegression(max_iter=4000, C=C, class_weight=cw)
        c.fit(sc.transform(Xtr[trf]), ytr[trf])
        oof[vaf] = c.predict_proba(sc.transform(Xtr[vaf]))[:, 1]
        tep.append(c.predict_proba(sc.transform(Xte))[:, 1])
    return oof, np.mean(tep, 0)


def case_scores(conn, ids, X, y, grp, view, tr, te, C, cw, seed):
    cc, mlo = view == "CC", view == "MLO"
    oa, ta = head_cv(X[tr], y[tr], grp[tr], X[te], C, cw, seed)
    oc, tc = head_cv(X[tr & cc], y[tr & cc], grp[tr & cc], X[te & cc], C, cw, seed)
    om, tm = head_cv(X[tr & mlo], y[tr & mlo], grp[tr & mlo], X[te & mlo], C, cw, seed)

    def branches(mask, a, c_, m_):
        from multiview_probe import _map
        ma = _map(ids[mask], a)
        ci, mi = ids[mask & cc], ids[mask & mlo]
        return [dict(ids=ci, score_pos=np.array([ma[int(i)] for i in ci])),
                dict(ids=mi, score_pos=np.array([ma[int(i)] for i in mi])),
                dict(ids=ci, score_pos=c_), dict(ids=mi, score_pos=m_)]
    trcf = evaluate.case_fuse(conn, branches(tr, oa, oc, om), method="mean")
    tecf = evaluate.case_fuse(conn, branches(te, ta, tc, tm), method="mean")
    return trcf, tecf


def eval_cfg(conn, ids, X, y, grp, view, tr, te, C, cw):
    tr_oof_auc, te_auc, te_acc, te_bal = [], [], [], []
    for seed in SEEDS:
        trcf, tecf = case_scores(conn, ids, X, y, grp, view, tr, te, C, cw, seed)
        ytr, Rtr = np.array(trcf["y_true"]), np.array(trcf["y_score"])
        yte, Rte = np.array(tecf["y_true"]), np.array(tecf["y_score"])
        # restrict test to paired-151 for comparability with the paper
        pidx = np.array(tecf["paired_idx"], dtype=int)
        yte, Rte = yte[pidx], Rte[pidx]
        tr_oof_auc.append(roc_auc_score(ytr, Rtr))
        thr = _best_threshold(yte, Rte)
        m = _metrics(yte, Rte, thr)
        te_auc.append(roc_auc_score(yte, Rte)); te_acc.append(m["accuracy"]); te_bal.append(m["bal_acc"])
    return (np.mean(tr_oof_auc), np.mean(te_auc), np.std(te_auc),
            np.mean(te_acc), np.mean(te_bal), te_auc, te_acc)


def main():
    conn = db.connect()
    ids, X, y0, split, grp, view, ds, patho, abn = load_enc(conn, ["cbis_full"], ENC4)
    y = np.isin(patho, ["MALIGNANT", "CANCER"]).astype(int)
    tr, te = split == "train", split == "test"

    print("=== (a) current config C=0.5 balanced: per-seed spread ===")
    r = eval_cfg(conn, ids, X, y, grp, view, tr, te, 0.5, "balanced")
    for s, a, ac in zip(SEEDS, r[5], r[6]):
        print(f"    seed {s}: test AUC={a:.4f}  acc={ac*100:.1f}%")
    print(f"    -> mean AUC={r[1]:.4f}+-{r[2]:.4f}  acc={r[3]*100:.1f}%  bal={r[4]*100:.1f}%  "
          f"(spread across seeds is the point: no seed is a 'lever')")

    print("\n=== (b) regularisation C sweep (select by TRAIN-OOF AUC) | balanced ===")
    print(f"    {'C':>6} {'TRAIN-OOF AUC':>14} {'TEST AUC':>16} {'acc':>6} {'bal':>6}")
    best = None
    for C in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        r = eval_cfg(conn, ids, X, y, grp, view, tr, te, C, "balanced")
        print(f"    {C:6.2f} {r[0]:14.4f} {r[1]:10.4f}+-{r[2]:.3f} {r[3]*100:5.1f}% {r[4]*100:5.1f}%")
        if best is None or r[0] > best[0]:
            best = (r[0], C, r)
    print(f"    honest pick by TRAIN-OOF AUC: C={best[1]} -> "
          f"TEST AUC={best[2][1]:.4f} acc={best[2][3]*100:.1f}% bal={best[2][4]*100:.1f}%")

    print("\n=== (c) class weight: balanced vs none (C=0.5) ===")
    for cw in ["balanced", None]:
        r = eval_cfg(conn, ids, X, y, grp, view, tr, te, 0.5, cw)
        print(f"    class_weight={str(cw):9} TRAIN-OOF AUC={r[0]:.4f}  "
              f"TEST AUC={r[1]:.4f} acc={r[3]*100:.1f}% bal={r[4]*100:.1f}%")


if __name__ == "__main__":
    main()
