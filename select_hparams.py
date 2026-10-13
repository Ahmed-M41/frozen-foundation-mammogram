"""Leakage-safe retrieval hyperparameter selection.

Picks (k, distance, head) using ONLY the gallery/train split, via patient-grouped
Stratified K-fold CV over the cached embeddings. The official test set is never
touched here. Selection metric = mean CV ROC-AUC (threshold-free, robust to class
imbalance), averaged over CC and MLO views and over folds.

This is NOT training — no parameters are learned. It only chooses among a small
grid of retrieval knobs, the honest analog of "tuning k on the training data".

Run:
    python select_hparams.py --embedder dinov2_vitl14            # report best
    python select_hparams.py --embedder dinov2_vitl14 --eval     # + honest test eval
"""
import argparse
import itertools

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import config
import db
import fusion
from retrieve import RetrievalIndex
from evaluate import compute_metrics, run_headline


def _load_train_view(conn, embedder, view, dataset="cbis"):
    ids, vecs, labels = db.load_embeddings(conn, embedder, split="train",
                                           view=view, dataset=dataset)
    if len(ids) == 0:
        return None
    meta = db.sample_meta(conn, ids)
    patients = np.array([meta[int(i)]["patient_id"] for i in ids])
    return ids, vecs, labels, patients


def cv_select(conn, embedder, dataset="cbis", n_splits=5, views=("CC", "MLO"),
              verbose=False):
    """Return (best_dict, ranked_list). best maximizes mean CV AUC."""
    data = {v: _load_train_view(conn, embedder, v, dataset) for v in views}
    data = {v: d for v, d in data.items() if d is not None}
    if not data:
        return None, []

    ranked = []
    for k, dist, head in itertools.product(config.K_GRID, config.DISTANCE_GRID,
                                           config.HEAD_GRID):
        per_view_auc = []
        for v, (ids, vecs, labels, patients) in data.items():
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=config.SEED)
            fold_aucs = []
            for tr_idx, va_idx in sgkf.split(vecs, labels, groups=patients):
                if len(set(labels[va_idx].tolist())) < 2:
                    continue
                index = RetrievalIndex(ids[tr_idx], vecs[tr_idx], labels[tr_idx],
                                       distance=dist)
                k_eff = min(k, len(tr_idx))
                sc, _, nl = index.search(vecs[va_idx], k_eff)
                _, sp = fusion.fuse(sc, nl, dist, head)
                m = compute_metrics(labels[va_idx], (sp >= 0.5).astype(int), sp)
                fold_aucs.append(m["auc"])
            if fold_aucs:
                per_view_auc.append(np.mean(fold_aucs))
        if per_view_auc:
            ranked.append(dict(cv_auc=float(np.mean(per_view_auc)), k=k,
                               distance=dist, head=head))
    ranked.sort(key=lambda r: r["cv_auc"], reverse=True)
    if verbose:
        for r in ranked[:5]:
            print(f"    cv_auc={r['cv_auc']:.4f}  k={r['k']} {r['distance']} {r['head']}")
    return (ranked[0] if ranked else None), ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=config.DEFAULT_EMBEDDER)
    ap.add_argument("--eval", action="store_true",
                    help="after selecting, evaluate that config on the official test set")
    ap.add_argument("--four-way", action="store_true")
    args = ap.parse_args()

    conn = db.connect()
    best, ranked = cv_select(conn, args.embedder, verbose=True)
    if best is None:
        print(f"RESULT: FAIL (no train embeddings for {args.embedder})")
        return
    print(f"  BEST (CV, train-only): k={best['k']} {best['distance']} {best['head']} "
          f"cv_auc={best['cv_auc']:.4f}")
    if args.eval:
        print("  --- honest test eval with CV-selected config ---")
        run_headline(conn, args.embedder, best["k"], best["distance"], best["head"],
                     four_way=args.four_way, variant="cv_selected")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
