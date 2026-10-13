"""Cross-dataset Family-1 experiments (4-way multi-view logistic probe on frozen
features). Three combinations the user asked for:

  A   train = CBIS-train + InBreast(all) ; test = CBIS-test
  B1  train = CBIS-train                 ; test = CBIS-test   (no InBreast at all)
  B2  train = CBIS-train                 ; test = InBreast    (zero-shot cross-dataset)

Everything on NATIVE full images (cbis_full / inbreast_full) so Mammo-CLIP sees its
proper input. Honest: 5-fold patient-grouped CV per head (oof sets the threshold),
fold-models averaged on the test set. CBIS test is case-level (paired reported);
InBreast has no CC/MLO pairing (anonymized) so its test is image-level.

Run:  python cross_probe.py --exp A
"""
import argparse

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import db
import evaluate
from mil import _best_threshold, _metrics

ENCODERS = ["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"]


def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def load_pool(conn, encoders, datasets):
    ids_l, X_l = [], []
    for ds in datasets:
        ids0, mats = None, []
        for e in encoders:
            ids, v, _ = db.load_embeddings(conn, e, dataset=ds)
            if len(ids) == 0:
                raise SystemExit(f"missing embeddings: {e} on {ds}")
            if ids0 is None:
                ids0 = ids
            else:
                assert np.array_equal(ids, ids0), f"misalign {e} {ds}"
            mats.append(_l2(v))
        ids_l.append(ids0)
        X_l.append(np.concatenate(mats, 1))
    ids = np.concatenate(ids_l)
    X = np.concatenate(X_l, 0).astype(np.float32)
    ph = ",".join("?" * len(datasets))
    info = {r["id"]: r for r in conn.execute(
        f"SELECT id,split,binary_label,patient_id,view,dataset FROM samples "
        f"WHERE dataset IN ({ph})", list(datasets))}
    c = lambda k: np.array([info[int(i)][k] for i in ids])
    return ids, X, c("binary_label"), c("split"), c("patient_id"), c("view"), c("dataset")


def head_cv(Xtr, ytr, ptr, Xte, C, seed=42, n_splits=5):
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof, tep = np.zeros(len(Xtr)), []
    for trf, vaf in sg.split(Xtr, ytr, groups=ptr):
        s = StandardScaler().fit(Xtr[trf])
        c = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        c.fit(s.transform(Xtr[trf]), ytr[trf])
        oof[vaf] = c.predict_proba(s.transform(Xtr[vaf]))[:, 1]
        tep.append(c.predict_proba(s.transform(Xte))[:, 1])
    return oof, np.mean(tep, axis=0)


def _branches(conn, ids, view, all_sc, cc_sc, mlo_sc):
    m_all = {int(i): float(s) for i, s in zip(ids, all_sc)}
    cc_ids, mlo_ids = ids[view == "CC"], ids[view == "MLO"]
    return evaluate.case_fuse(conn, [
        dict(ids=cc_ids, score_pos=np.array([m_all[int(i)] for i in cc_ids])),
        dict(ids=mlo_ids, score_pos=np.array([m_all[int(i)] for i in mlo_ids])),
        dict(ids=cc_ids, score_pos=cc_sc),
        dict(ids=mlo_ids, score_pos=mlo_sc),
    ], method="mean")


EXPS = {
    "A":  (["cbis_full", "inbreast_full"],
           lambda ds, sp: ((ds == "cbis_full") & (sp == "train")) | (ds == "inbreast_full"),
           lambda ds, sp: (ds == "cbis_full") & (sp == "test")),
    "B1": (["cbis_full"],
           lambda ds, sp: (ds == "cbis_full") & (sp == "train"),
           lambda ds, sp: (ds == "cbis_full") & (sp == "test")),
    "B2": (["cbis_full", "inbreast_full"],
           lambda ds, sp: (ds == "cbis_full") & (sp == "train"),
           lambda ds, sp: (ds == "inbreast_full")),
}


def run(exp, C=0.05, test_size=0.3, seed=42):
    from sklearn.model_selection import train_test_split
    conn = db.connect()
    if exp == "C":                 # InBreast-only, image-level stratified split (no official split)
        ids, X, y, split, pat, view, ds = load_pool(conn, ENCODERS, ["inbreast_full"])
        tr_idx, _ = train_test_split(np.arange(len(ids)), test_size=test_size,
                                     stratify=y, random_state=seed)
        tr = np.zeros(len(ids), bool); tr[tr_idx] = True
        te = ~tr
    else:
        datasets, train_sel, test_sel = EXPS[exp]
        ids, X, y, split, pat, view, ds = load_pool(conn, ENCODERS, datasets)
        tr, te = train_sel(ds, split), test_sel(ds, split)
    cc, mlo = view == "CC", view == "MLO"

    oof_all, te_all = head_cv(X[tr], y[tr], pat[tr], X[te], C)
    oof_cc, te_cc = head_cv(X[tr & cc], y[tr & cc], pat[tr & cc], X[te & cc], C)
    oof_mlo, te_mlo = head_cv(X[tr & mlo], y[tr & mlo], pat[tr & mlo], X[te & mlo], C)

    tr_cf = _branches(conn, ids[tr], view[tr], oof_all, oof_cc, oof_mlo)
    thr = _best_threshold(tr_cf["y_true"], tr_cf["y_score"])
    cv_auc = roc_auc_score(tr_cf["y_true"], tr_cf["y_score"])
    cf = _branches(conn, ids[te], view[te], te_all, te_cc, te_mlo)
    yv, sv, pidx = cf["y_true"], cf["y_score"], cf["paired_idx"]
    full = _metrics(yv, sv, thr)
    print(f"=== EXP {exp}: n_tr={tr.sum()} n_te={len(yv)} "
          f"| C={C} thr={thr:.3f} train-CV-AUC={cv_auc:.4f}")
    print(f"    TEST({len(yv)}): acc={full['accuracy']:.4f} bal_acc={full['bal_acc']:.4f} "
          f"auc={full['auc']:.4f} sens={full['sensitivity']:.4f} spec={full['specificity']:.4f}")
    if len(pidx) > 0:
        pm = _metrics(yv[pidx], sv[pidx], thr)
        print(f"    PAIRED({len(pidx)}): acc={pm['accuracy']:.4f} bal_acc={pm['bal_acc']:.4f} "
              f"auc={pm['auc']:.4f} sens={pm['sensitivity']:.4f} spec={pm['specificity']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["A", "B1", "B2", "C"])
    ap.add_argument("--C", type=float, default=0.05)
    args = ap.parse_args()
    run(args.exp, C=args.C)
