"""'Pair image' = EARLY (feature) fusion of the two views into ONE input.

For each PAIRED breast case, concatenate the CC and MLO frozen features into a
single vector [cc_feats | mlo_feats] and train ONE logistic head on it — i.e. the
model sees both views together as a single entity, vs. our 4-way multi-view which
classifies each view separately and averages the scores ('pair of images').

Paired cases only (both views present). Honest 5-fold patient-grouped CV, tuned
threshold, best-C. Compares directly to the 4-way multi-view on the same 151.

Run:  python pair_image.py --encoders mammo_clip_b5 dinov2_vitl14_noclahe rad_dino_noclahe
"""
import argparse

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import db
from mil import _best_threshold, _metrics
from twoview_wholeimage import load_cases


def run(encoders, dataset="cbis_full", C=0.05, seed=42, n_splits=5):
    conn = db.connect()
    Xcc, Xmlo, y, split, pat, paired = load_cases(conn, encoders, dataset)
    Xpair = np.concatenate([Xcc, Xmlo], axis=1)      # [n_cases, 2*D] — one paired vector
    tr = (split == "train") & paired
    te = (split == "test") & paired
    Xtr, ytr, ptr = Xpair[tr], y[tr], pat[tr]
    Xte, yte = Xpair[te], y[te]

    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof, tep = np.zeros(len(Xtr)), []
    for trf, vaf in sg.split(Xtr, ytr, groups=ptr):
        s = StandardScaler().fit(Xtr[trf])
        c = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        c.fit(s.transform(Xtr[trf]), ytr[trf])
        oof[vaf] = c.predict_proba(s.transform(Xtr[vaf]))[:, 1]
        tep.append(c.predict_proba(s.transform(Xte))[:, 1])
    te_score = np.mean(tep, axis=0)
    thr = _best_threshold(ytr, oof)
    m = _metrics(yte, te_score, thr)
    print(f"  PairImage(early-fusion)[{'+'.join(encoders)}] C={C} thr={thr:.3f} "
          f"CV-AUC={roc_auc_score(ytr, oof):.4f} n_tr={tr.sum()} n_te={te.sum()}")
    print(f"    PAIRED({te.sum()}): acc={m['accuracy']:.4f} bal_acc={m['bal_acc']:.4f} "
          f"auc={m['auc']:.4f} sens={m['sensitivity']:.4f} spec={m['specificity']:.4f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+",
                    default=["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"])
    ap.add_argument("--dataset", default="cbis_full")
    ap.add_argument("--C", type=float, default=0.05)
    args = ap.parse_args()
    run(args.encoders, dataset=args.dataset, C=args.C)
