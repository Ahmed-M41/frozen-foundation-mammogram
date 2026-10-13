"""Multi-view linear probe on frozen Mammo-CLIP features — the faithful analog of
the old 4-model ensemble, but with logistic heads on frozen features (no fine-tune).

Three heads on the frozen features:
  CC-specialist  (trained on CC images only)
  MLO-specialist (trained on MLO images only)
  ALL            (trained on all images)

Fusion modes (per breast case):
  allonly  = (ALL·cc + ALL·mlo)/2                     [what linear_probe.py did]
  spec2    = (CC·cc + MLO·mlo)/2                       [view specialists only]
  fourway  = (CC·cc + ALL·cc + MLO·mlo + ALL·mlo)/4    [= old triple_weighted]

Honest: 5-fold patient-grouped CV per head (out-of-fold sets the threshold; the
fold-models are averaged on the official test set). Case-level metrics (full + paired).

Run:  python multiview_probe.py --encoders mammo_clip_b5 --dataset cbis_full --mode fourway
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


def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def load(conn, encoders, dataset):
    ids0, mats = None, []
    for e in encoders:
        ids, v, _ = db.load_embeddings(conn, e, dataset=dataset)
        if ids0 is None:
            ids0 = ids
        else:
            assert np.array_equal(ids, ids0), f"misalign {e}"
        mats.append(_l2(v))
    X = np.concatenate(mats, 1).astype(np.float32)
    info = {r["id"]: r for r in conn.execute(
        "SELECT id,split,binary_label,patient_id,view FROM samples WHERE dataset=?", (dataset,))}
    split = np.array([info[int(i)]["split"] for i in ids0])
    y = np.array([info[int(i)]["binary_label"] for i in ids0])
    pat = np.array([info[int(i)]["patient_id"] for i in ids0])
    view = np.array([info[int(i)]["view"] for i in ids0])
    return np.asarray(ids0), X, y, split, pat, view


# --- Tier-2 toggles (0 = off = baseline). ---
FROFA = 0            # number of augmented copies of the training features
FROFA_SIGMA = 0.1    # per-feature multiplicative jitter scale
PCA_DIM = 0          # if >0, PCA-reduce features to this many dims before the head


def _fit_tf(X):
    s = StandardScaler().fit(X)
    Xt = s.transform(X)
    pca = None
    if PCA_DIM > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(PCA_DIM, X.shape[1], X.shape[0]), random_state=0).fit(Xt)
        Xt = pca.transform(Xt)
    return (s, pca), Xt


def _apply_tf(tf, X):
    s, pca = tf
    Xt = s.transform(X)
    return pca.transform(Xt) if pca is not None else Xt


def head_cv(Xtr, ytr, ptr, Xte, C, seed=42, n_splits=5):
    """Return (oof scores on train images, ensemble scores on test images)."""
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof, tep = np.zeros(len(Xtr)), []
    rng = np.random.default_rng(seed)
    for trf, vaf in sg.split(Xtr, ytr, groups=ptr):
        Xf, yf = Xtr[trf], ytr[trf]
        if FROFA > 0:                                  # augment ONLY the fit set
            px, py = [Xf], [yf]
            for _ in range(FROFA):
                px.append(Xf * (1.0 + FROFA_SIGMA * rng.standard_normal(Xf.shape)))
                py.append(yf)
            Xf, yf = np.concatenate(px), np.concatenate(py)
        tf, Xf_t = _fit_tf(Xf)
        c = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        c.fit(Xf_t, yf)
        oof[vaf] = c.predict_proba(_apply_tf(tf, Xtr[vaf]))[:, 1]   # eval on ORIGINAL val
        tep.append(c.predict_proba(_apply_tf(tf, Xte))[:, 1])
    return oof, np.mean(tep, axis=0)


def head_full(Xtr, ytr, ptr, Xte, C, seed=42, n_splits=5):
    """Full-data head (matches the headline eval): fit ONE model on all train,
    score train IN-SAMPLE and test. Same signature as head_cv so it's a drop-in."""
    tf, Xtr_t = _fit_tf(Xtr)
    c = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
    c.fit(Xtr_t, ytr)
    tr = c.predict_proba(Xtr_t)[:, 1]
    te = c.predict_proba(_apply_tf(tf, Xte))[:, 1]
    return tr, te


def _map(ids, scores):
    return {int(i): float(s) for i, s in zip(ids, scores)}


def run(encoders, dataset="cbis_full", mode="fourway", C=0.05):
    conn = db.connect()
    ids, X, y, split, pat, view = load(conn, encoders, dataset)
    tr, te = split == "train", split == "test"
    cc, mlo = view == "CC", view == "MLO"

    # train the three heads (oof on train, ensemble on test)
    oof_all, te_all = head_cv(X[tr], y[tr], pat[tr], X[te], C)
    oof_cc, te_cc = head_cv(X[tr & cc], y[tr & cc], pat[tr & cc], X[te & cc], C)
    oof_mlo, te_mlo = head_cv(X[tr & mlo], y[tr & mlo], pat[tr & mlo], X[te & mlo], C)

    def branches(mask, all_scores, cc_scores, mlo_scores):
        ids_all = ids[mask]
        m_all = _map(ids_all, all_scores)
        cc_ids, mlo_ids = ids[mask & cc], ids[mask & mlo]
        br = []
        if mode in ("allonly", "fourway"):
            br.append(dict(ids=cc_ids, score_pos=np.array([m_all[int(i)] for i in cc_ids])))
            br.append(dict(ids=mlo_ids, score_pos=np.array([m_all[int(i)] for i in mlo_ids])))
        if mode in ("spec2", "fourway"):
            br.append(dict(ids=cc_ids, score_pos=cc_scores))
            br.append(dict(ids=mlo_ids, score_pos=mlo_scores))
        return br

    thr_cf = evaluate.case_fuse(conn, branches(tr, oof_all, oof_cc, oof_mlo), method="mean")
    thr = _best_threshold(thr_cf["y_true"], thr_cf["y_score"])
    cv_auc = roc_auc_score(thr_cf["y_true"], thr_cf["y_score"])   # honest combo-selection signal
    cf = evaluate.case_fuse(conn, branches(te, te_all, te_cc, te_mlo), method="mean")
    y_te, s_te, pidx = cf["y_true"], cf["y_score"], cf["paired_idx"]
    full = _metrics(y_te, s_te, thr)
    pm = _metrics(y_te[pidx], s_te[pidx], thr)
    run.last_cf = cf              # expose TEST case scores for fusion
    run.last_train_cf = thr_cf    # expose TRAIN out-of-fold case scores (for honest threshold)
    print(f"  MultiViewProbe[{'+'.join(encoders)}] mode={mode} C={C} thr={thr:.3f} "
          f"TRAIN-CV-case-AUC={cv_auc:.4f}")
    print(f"    FULL({len(y_te)}):   acc={full['accuracy']:.4f} bal_acc={full['bal_acc']:.4f} "
          f"auc={full['auc']:.4f} sens={full['sensitivity']:.4f} spec={full['specificity']:.4f}")
    print(f"    PAIRED({len(pidx)}): acc={pm['accuracy']:.4f} bal_acc={pm['bal_acc']:.4f} "
          f"auc={pm['auc']:.4f} sens={pm['sensitivity']:.4f} spec={pm['specificity']:.4f}")
    return full, pm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["mammo_clip_b5"])
    ap.add_argument("--dataset", default="cbis_full")
    ap.add_argument("--mode", default="fourway", choices=["allonly", "spec2", "fourway"])
    ap.add_argument("--C", type=float, default=0.05)
    args = ap.parse_args()
    run(args.encoders, dataset=args.dataset, mode=args.mode, C=args.C)
