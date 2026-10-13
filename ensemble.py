"""Multi-encoder ensembles over CACHED embeddings (no re-embedding), evaluated
at case level (CC+MLO fused).

  concat (early): L2-normalize each encoder's vectors and concatenate into one
                  joint retrieval space, per view -> kNN + fusion -> case fuse.
  late   (soft) : classify each (encoder, view) branch independently, then let
                  evaluate.case_fuse average all per-image scores per breast case.

All encoders share the samples/split and db.load_embeddings orders by sample id,
so per-encoder arrays are aligned by construction (asserted). Training-free.
"""
import itertools

import numpy as np

import config
import db
import fusion
import evaluate
from retrieve import RetrievalIndex


def _l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def _load_view(conn, encoders, split, view, dataset="cbis"):
    ids0, labels0, mats = None, None, []
    for e in encoders:
        ids, vecs, labels = db.load_embeddings(conn, e, split=split, view=view,
                                               dataset=dataset)
        if len(ids) == 0:
            return None, None, None
        if ids0 is None:
            ids0, labels0 = ids, labels
        else:
            assert np.array_equal(ids, ids0), f"sample misalignment for {e}"
        mats.append(_l2(vecs))
    return ids0, labels0, np.concatenate(mats, axis=1)


def concat_branches(conn, encoders, k, distance, head, dataset="cbis"):
    branches = []
    for view in config.VIEWS:
        tr_ids, tr_lab, tr = _load_view(conn, encoders, "train", view, dataset)
        te_ids, te_lab, te = _load_view(conn, encoders, "test", view, dataset)
        if tr is None or te is None:
            continue
        index = RetrievalIndex(tr_ids, tr, tr_lab, distance=distance)
        leak = set(np.asarray(te_ids).tolist()) & set(index.train_ids.tolist())
        assert not leak, f"DATA LEAK in ensemble: {len(leak)} ids"
        k_eff = min(k, len(tr_ids))
        sc, _, nl = index.search(te, k_eff)
        pred, sp = fusion.fuse(sc, nl, distance, head)
        branches.append(dict(ids=np.asarray(te_ids), labels=np.asarray(te_lab),
                             pred=pred, score_pos=sp, view=view))
    return branches


def late_branches(conn, encoders, k, distance, head, dataset="cbis"):
    """One branch per (encoder, view); case_fuse averages them per case."""
    branches = []
    for e in encoders:
        for view in config.VIEWS:
            res = evaluate.image_branch(conn, e, k, distance, head, view, view, dataset)
            if res is not None:
                branches.append(res)
    return branches


def run_ensemble(conn, encoders, k, distance, head, mode="concat", dataset="cbis",
                 dataset_scope="cbis_mass", variant=None, persist=True, verbose=True):
    variant = variant or f"ens_{mode}"
    if mode == "concat":
        branches = concat_branches(conn, encoders, k, distance, head, dataset)
    elif mode == "late":
        branches = late_branches(conn, encoders, k, distance, head, dataset)
    else:
        raise ValueError(mode)
    if not branches:
        if verbose:
            print(f"  [skip] ensemble {encoders}: missing embeddings")
        return None

    cf = evaluate.case_fuse(conn, branches)
    label = f"ens_{mode}[" + "+".join(encoders) + "]"
    if persist:
        evaluate.persist_experiment(conn, label, k, distance, head, "CC+MLO",
                                    "both", "case", dataset_scope, cf["n_cases"],
                                    cf["metrics"], variant)
    if verbose:
        m = cf["metrics"]
        pm = cf["paired_metrics"] or {}
        print(f"  [{label} k={k} {distance} {head}] n={cf['n_cases']} "
              f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f} "
              f"sens={m['sensitivity']:.4f} spec={m['specificity']:.4f} "
              f"| paired auc={pm.get('auc', 0):.4f}")
    return dict(label=label, case=cf)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=config.DEFAULT_K)
    ap.add_argument("--distance", default=config.DEFAULT_DISTANCE)
    ap.add_argument("--head", default=config.DEFAULT_HEAD)
    ap.add_argument("--mode", default="concat", choices=["concat", "late"])
    args = ap.parse_args()
    conn = db.connect()
    res = run_ensemble(conn, args.encoders, args.k, args.distance, args.head, mode=args.mode)
    print("RESULT:", "PASS" if res else "FAIL")
