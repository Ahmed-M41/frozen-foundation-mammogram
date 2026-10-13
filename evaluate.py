"""Stage 4c: retrieval + fusion + metrics, at image-level and case-level.

An "image branch" = build a gallery index (train, filtered by view) and classify
the query images (test, filtered by view) via kNN + fusion. The headline result
fuses the CC branch and the MLO branch per breast case (a training-free analog of
the prior project's multi-view ensemble).

Everything here is training-free: no gradients, no learned head. Retrieval knobs
(k, distance, head) are selected on the gallery only (see select_hparams.py), so
the official test set stays untouched.
"""
import json

import numpy as np
from collections import defaultdict
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)

import config
import db
import retrieve
import fusion


def compute_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    auc = 0.0
    if len(set(y_true.tolist())) > 1:
        auc = roc_auc_score(y_true, y_score)
    return dict(accuracy=float(acc), f1=float(f1), auc=float(auc),
                sensitivity=float(sens), specificity=float(spec),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def image_branch(conn, embedder, k, distance, head, gallery_view, query_view,
                 dataset="cbis"):
    """One retrieval branch. gallery_view/query_view in {'CC','MLO','all'}.
    Returns per-image predictions + metrics, or None if data is missing."""
    g_view = None if gallery_view == "all" else gallery_view
    q_view = None if query_view == "all" else query_view
    index = retrieve.build_index(conn, embedder, distance, view=g_view, dataset=dataset)
    ids, vecs, labels = db.load_embeddings(conn, embedder, split="test",
                                           view=q_view, dataset=dataset)
    if index is None or len(ids) == 0:
        return None

    leak = set(np.asarray(ids).tolist()) & set(index.train_ids.tolist())
    assert not leak, f"DATA LEAK: {len(leak)} test ids present in gallery"

    k_eff = min(k, len(index.train_ids))
    scores, nbr_ids, nbr_labels = index.search(vecs, k_eff)
    pred, score_pos = fusion.fuse(scores, nbr_labels, distance, head)
    m = compute_metrics(labels, pred, score_pos)
    return dict(ids=np.asarray(ids), labels=np.asarray(labels), pred=pred,
                score_pos=score_pos, nbr_ids=nbr_ids, scores=scores,
                k_eff=k_eff, gallery_view=gallery_view, query_view=query_view,
                metrics=m)


def _combine_views(vals, method):
    """Combine the per-view malignant scores of one breast case into one score.
    This is the CC+MLO fusion that lifted the prior work from ~78/79% to 83%."""
    a = np.asarray(vals, dtype=float)
    if method == "mean":
        return float(a.mean())
    if method == "max":                      # cancer visible in either view -> flag
        return float(a.max())
    if method == "min":                      # both views must agree -> specificity
        return float(a.min())
    if method in ("product", "noisy_or"):    # 1 - prod(1 - p): sensitivity-boosting
        return float(1.0 - np.prod(1.0 - a))
    raise ValueError(f"unknown view-fusion method {method!r}")


def case_fuse(conn, branches, method="mean"):
    """Aggregate per-image malignant scores into one score per breast case.

    Per image: mean of its score across branches. Per case: combine the case's
    per-view scores (CC and MLO) via `method` (mean/max/min/noisy_or). Returns
    overall metrics plus metrics on the 'paired' subset (cases having BOTH a CC
    and an MLO query image)."""
    img_scores = defaultdict(list)
    for b in branches:
        for sid, sc in zip(b["ids"], b["score_pos"]):
            img_scores[int(sid)].append(float(sc))
    meta = db.sample_meta(conn, list(img_scores.keys()))

    case_img_score = defaultdict(list)   # case -> [per-image mean score]
    case_views = defaultdict(set)
    case_label = defaultdict(int)
    for sid, scs in img_scores.items():
        m = meta[sid]
        cid = m["case_id"]
        case_img_score[cid].append(float(np.mean(scs)))
        case_views[cid].add(m["view"])
        case_label[cid] = max(case_label[cid], int(m["binary_label"]))

    cids = sorted(case_img_score)
    y_score = np.array([_combine_views(case_img_score[c], method) for c in cids])
    y_true = np.array([case_label[c] for c in cids])
    y_pred = (y_score >= 0.5).astype(np.int64)
    overall = compute_metrics(y_true, y_pred, y_score)

    paired = [i for i, c in enumerate(cids) if len(case_views[c]) >= 2]
    paired_metrics = None
    if paired:
        pi = np.array(paired)
        paired_metrics = compute_metrics(y_true[pi], y_pred[pi], y_score[pi])
    return dict(cids=cids, y_true=y_true, y_pred=y_pred, y_score=y_score,
                metrics=overall, n_cases=len(cids), n_paired=len(paired),
                paired_idx=np.array(paired, dtype=int), paired_metrics=paired_metrics)


# --- persistence -------------------------------------------------------------

def persist_experiment(conn, embedder, k, distance, head, gallery_view,
                       query_view, level, dataset_scope, n_test, m, variant):
    seed = int(db.get_meta(conn, "seed", config.SEED))
    cur = conn.execute(
        "INSERT INTO experiments (embedder_name,k,distance,head,gallery_view,"
        "query_view,level,dataset_scope,variant,seed,n_test,accuracy,f1,auc,"
        "sensitivity,specificity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (embedder, k, distance, head, gallery_view, query_view, level,
         dataset_scope, variant, seed, n_test, m["accuracy"], m["f1"],
         m["auc"], m["sensitivity"], m["specificity"]))
    conn.commit()
    return cur.lastrowid


def run_headline(conn, embedder, k, distance, head, dataset="cbis",
                 dataset_scope="cbis_mass", four_way=False, variant="baseline",
                 persist=True, verbose=True, save_predictions=True):
    """Headline config: CC branch + MLO branch (+ optional view-agnostic branches),
    fused per case. Persists image-level (CC, MLO) and case-level experiment rows."""
    branch_specs = [("CC", "CC"), ("MLO", "MLO")]
    if four_way:
        branch_specs += [("all", "CC"), ("all", "MLO")]

    branches = []
    for gv, qv in branch_specs:
        res = image_branch(conn, embedder, k, distance, head, gv, qv, dataset)
        if res is None:
            continue
        branches.append(res)
        if persist:
            persist_experiment(conn, embedder, res["k_eff"], distance, head, gv,
                               qv, "image", dataset_scope, len(res["ids"]),
                               res["metrics"], variant)
        if verbose:
            m = res["metrics"]
            print(f"  [img {gv}->{qv} k={res['k_eff']} {distance} {head}] "
                  f"n={len(res['ids'])} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                  f"auc={m['auc']:.4f} sens={m['sensitivity']:.4f} spec={m['specificity']:.4f}")

    if not branches:
        if verbose:
            print("  [skip] no branches produced data")
        return None

    cf = case_fuse(conn, branches)
    if persist:
        exp_id = persist_experiment(conn, embedder, k, distance, head,
                                    "CC+MLO" if not four_way else "all4", "both",
                                    "case", dataset_scope, cf["n_cases"],
                                    cf["metrics"], variant)
        if save_predictions:
            for i, cid in enumerate(cf["cids"]):
                conn.execute(
                    "INSERT INTO predictions (experiment_id,case_id,predicted_label,"
                    "true_label,score_pos,correct) VALUES (?,?,?,?,?,?)",
                    (exp_id, cid, int(cf["y_pred"][i]), int(cf["y_true"][i]),
                     float(cf["y_score"][i]),
                     int(cf["y_pred"][i] == cf["y_true"][i])))
            conn.commit()
    if verbose:
        m = cf["metrics"]
        print(f"  [CASE {'all4' if four_way else 'CC+MLO'} k={k} {distance} {head}] "
              f"n_cases={cf['n_cases']} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"auc={m['auc']:.4f} sens={m['sensitivity']:.4f} spec={m['specificity']:.4f}")
        if cf["paired_metrics"]:
            pm = cf["paired_metrics"]
            print(f"        paired-only (n={cf['n_paired']}): acc={pm['accuracy']:.4f} "
                  f"f1={pm['f1']:.4f} auc={pm['auc']:.4f}")
    return dict(branches=branches, case=cf)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=config.DEFAULT_EMBEDDER)
    ap.add_argument("--k", type=int, default=config.DEFAULT_K)
    ap.add_argument("--distance", default=config.DEFAULT_DISTANCE)
    ap.add_argument("--head", default=config.DEFAULT_HEAD)
    ap.add_argument("--four-way", action="store_true")
    args = ap.parse_args()
    conn = db.connect()
    res = run_headline(conn, args.embedder, args.k, args.distance, args.head,
                       four_way=args.four_way)
    print("RESULT:", "PASS" if res is not None else "FAIL")
