"""Family 1, Option 3: a trained logistic head on FROZEN whole-image features.

Simplest trained-head baseline (backbone frozen). Same honest protocol as the MIL:
5-fold patient-grouped CV on train (out-of-fold sets the threshold; 5 fold-models
averaged on the official test set), per-image scores fused per breast case, then
case-level metrics (full + paired). Lets us compare Option 3 vs Options 1/2 fairly.

Run:  python linear_probe.py --encoders mammo_clip_b5 --dataset cbis_full
      python linear_probe.py --encoders dinov2_vitb14_noclahe dinov2_vitl14_noclahe rad_dino_noclahe
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

import db
import evaluate
from mil import _best_threshold, _metrics


def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def load_concat(conn, encoders, split, dataset):
    ids0, lab0, mats = None, None, []
    for e in encoders:
        ids, v, lab = db.load_embeddings(conn, e, split=split, view=None, dataset=dataset)
        if ids0 is None:
            ids0, lab0 = ids, lab
        else:
            assert np.array_equal(ids, ids0), f"misalign {e}"
        mats.append(_l2(v))
    return ids0, np.concatenate(mats, 1), lab0


def _patients(conn, ids):
    meta = db.sample_meta(conn, ids)
    return np.array([meta[int(i)]["patient_id"] for i in ids])


def run(conn, encoders, dataset="cbis", method="mean", C=0.05, seed=42, n_splits=5):
    idtr, Xtr, ytr = load_concat(conn, encoders, "train", dataset)
    idte, Xte, yte = load_concat(conn, encoders, "test", dataset)
    ptr = _patients(conn, idtr)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(Xtr))
    te_preds = []
    for trf, vaf in sgkf.split(Xtr, ytr, groups=ptr):
        sc = StandardScaler().fit(Xtr[trf])
        clf = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        clf.fit(sc.transform(Xtr[trf]), ytr[trf])
        oof[vaf] = clf.predict_proba(sc.transform(Xtr[vaf]))[:, 1]
        te_preds.append(clf.predict_proba(sc.transform(Xte))[:, 1])
    te_score = np.mean(te_preds, axis=0)

    cf_tr = evaluate.case_fuse(conn, [dict(ids=idtr, score_pos=oof)], method=method)
    thr = _best_threshold(cf_tr["y_true"], cf_tr["y_score"])
    cf = evaluate.case_fuse(conn, [dict(ids=idte, score_pos=te_score)], method=method)
    y, s, pidx = cf["y_true"], cf["y_score"], cf["paired_idx"]
    full = _metrics(y, s, thr)
    pm = _metrics(y[pidx], s[pidx], thr)
    print(f"  LinearProbe[{'+'.join(encoders)}] C={C} method={method} thr={thr:.3f}")
    print(f"    FULL({len(y)}):   acc={full['accuracy']:.4f} bal_acc={full['bal_acc']:.4f} "
          f"auc={full['auc']:.4f} sens={full['sensitivity']:.4f} spec={full['specificity']:.4f}")
    print(f"    PAIRED({len(pidx)}): acc={pm['accuracy']:.4f} bal_acc={pm['bal_acc']:.4f} "
          f"auc={pm['auc']:.4f} sens={pm['sensitivity']:.4f} spec={pm['specificity']:.4f}")
    return full, pm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--dataset", default="cbis")
    ap.add_argument("--method", default="mean", choices=["mean", "max", "min", "noisy_or"])
    ap.add_argument("--C", type=float, default=0.05)
    args = ap.parse_args()
    run(db.connect(), args.encoders, dataset=args.dataset, method=args.method, C=args.C)
