"""Single-image (one-view) evaluation — NO CC/MLO fusion.

Trains one logistic head on the frozen features and scores each test IMAGE on its
own (image-level metrics), broken out by view. This is the per-image baseline that
the case-level 4-way multi-view (0.795) is built on top of — the gap between them
is exactly what multi-view fusion buys.

Run:  python single_image.py --encoders mammo_clip_b5 dinov2_vitl14_noclahe rad_dino_noclahe
"""
import argparse
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import db
from mil import _best_threshold, _metrics
from linear_probe import load_concat, _patients


def run(encoders, dataset="cbis_full", C=0.05, seed=42, n_splits=5):
    conn = db.connect()
    idtr, Xtr, ytr = load_concat(conn, encoders, "train", dataset)
    idte, Xte, yte = load_concat(conn, encoders, "test", dataset)
    ptr = _patients(conn, idtr)
    meta = db.sample_meta(conn, list(idte))
    view = np.array([meta[int(i)]["view"] for i in idte])
    case = np.array([meta[int(i)]["case_id"] for i in idte])
    cv = defaultdict(set)
    for c, v in zip(case, view):
        cv[c].add(v)
    paired_cases = {c for c, vs in cv.items() if len(vs) >= 2}
    paired_img = np.array([c in paired_cases for c in case])

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

    print(f"  SingleImage[{'+'.join(encoders)}] C={C} thr={thr:.3f}  (image-level, NO fusion)")

    def rep(name, mask):
        if mask.sum() == 0:
            return
        m = _metrics(yte[mask], te_score[mask], thr)
        print(f"    {name:24s} (n={int(mask.sum())}): acc={m['accuracy']:.4f} "
              f"bal_acc={m['bal_acc']:.4f} auc={m['auc']:.4f} "
              f"sens={m['sensitivity']:.4f} spec={m['specificity']:.4f}")

    rep("ALL test images", np.ones(len(yte), bool))
    rep("CC images only", view == "CC")
    rep("MLO images only", view == "MLO")
    rep("paired-cases images", paired_img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+",
                    default=["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"])
    ap.add_argument("--dataset", default="cbis_full")
    ap.add_argument("--C", type=float, default=0.05)
    args = ap.parse_args()
    run(args.encoders, dataset=args.dataset, C=args.C)
