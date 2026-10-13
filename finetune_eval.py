"""Reviewer-response control: score the fine-tuned Mammo-CLIP backbone through the
EXACT SAME multi-view linear-probe structure multiview_probe.py uses for the frozen
encoder (three heads: CC-specialist / MLO-specialist / ALL, case-level fourway
fusion), so the two numbers are directly comparable -- only the encoder differs.

Requires finetune_mammo_clip.py to have already produced embedder_names
mammo_clip_b5_ft_fold0..4 and cache/finetune_fold_assignment.json.

Fold-matched features: image X in outer fold V is scored using embeddings from
mammo_clip_b5_ft_fold{V} -- the ONE backbone that never saw fold V's labels during
its own fine-tuning -- for BOTH the "fit" images (folds != V, in-sample to that
backbone but that's fine, we're not evaluating them here) AND the "val" images
(fold V, honestly out-of-fold). This is the nested-CV discipline that keeps the
comparison leak-free; see finetune_mammo_clip.py's module docstring for the full
argument. It mirrors multiview_probe.head_cv's oof/tep pattern one level down.

Run: python finetune_eval.py --mode fourway
     (also prints the frozen mammo_clip_b5 baseline via multiview_probe.py for a
      direct side-by-side, and the fine-tuning wall-clock/GPU cost from
      cache/finetune_log.json)
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
import db
import evaluate
from mil import _best_threshold, _metrics
from multiview_probe import _l2, _fit_tf, _apply_tf, run as run_frozen

CACHE = os.path.join(config.PROJECT_DIR, "cache")
FOLD_JSON = os.path.join(CACHE, "finetune_fold_assignment.json")
LOG_JSON = os.path.join(CACHE, "finetune_log.json")
DATASET = "cbis_full"
N_SPLITS = 5


def _load_fold_of():
    with open(FOLD_JSON) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _meta(conn):
    rows = conn.execute(
        "SELECT id,split,binary_label,patient_id,view FROM samples WHERE dataset=?",
        (DATASET,)).fetchall()
    return {r["id"]: r for r in rows}


def head_finetuned(conn, fold_of, info, C, view_filter=None):
    """Fold-matched analog of multiview_probe.head_cv. Returns (oof_map, test_map),
    each {sample_id: score}, same contract as multiview_probe's _map()."""
    oof, test_scores = {}, {}
    for V in range(N_SPLITS):
        name = f"mammo_clip_b5_ft_fold{V}"
        ids, X, _ = db.load_embeddings(conn, name, dataset=DATASET)
        X = _l2(X)
        if view_filter is not None:
            keep = np.array([info[int(i)]["view"] == view_filter for i in ids])
            ids, X = ids[keep], X[keep]

        is_train = np.array([info[int(i)]["split"] == "train" for i in ids])
        is_test = np.array([info[int(i)]["split"] == "test" for i in ids])
        tr_ids, Xtr = ids[is_train], X[is_train]
        te_ids, Xte = ids[is_test], X[is_test]

        fold_v = np.array([fold_of[int(i)] for i in tr_ids])
        fit_mask, val_mask = fold_v != V, fold_v == V
        if fit_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        ytr = np.array([info[int(i)]["binary_label"] for i in tr_ids])
        tf, Xf_t = _fit_tf(Xtr[fit_mask])
        clf = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        clf.fit(Xf_t, ytr[fit_mask])

        val_pred = clf.predict_proba(_apply_tf(tf, Xtr[val_mask]))[:, 1]
        for sid, s in zip(tr_ids[val_mask], val_pred):
            oof[int(sid)] = float(s)

        te_pred = clf.predict_proba(_apply_tf(tf, Xte))[:, 1]
        for sid, s in zip(te_ids, te_pred):
            test_scores.setdefault(int(sid), []).append(float(s))

    test_scores = {sid: float(np.mean(v)) for sid, v in test_scores.items()}
    return oof, test_scores


def _branches(ids_subset, cc_ids, mlo_ids, all_map, cc_map, mlo_map, mode):
    br = []
    if mode in ("allonly", "fourway"):
        br.append(dict(ids=cc_ids, score_pos=np.array([all_map[int(i)] for i in cc_ids])))
        br.append(dict(ids=mlo_ids, score_pos=np.array([all_map[int(i)] for i in mlo_ids])))
    if mode in ("spec2", "fourway"):
        br.append(dict(ids=cc_ids, score_pos=np.array([cc_map[int(i)] for i in cc_ids])))
        br.append(dict(ids=mlo_ids, score_pos=np.array([mlo_map[int(i)] for i in mlo_ids])))
    return br


def run(mode="fourway", C=0.05):
    conn = db.connect()
    fold_of = _load_fold_of()
    info = _meta(conn)

    oof_all, te_all = head_finetuned(conn, fold_of, info, C, view_filter=None)
    oof_cc, te_cc = head_finetuned(conn, fold_of, info, C, view_filter="CC")
    oof_mlo, te_mlo = head_finetuned(conn, fold_of, info, C, view_filter="MLO")

    tr_ids = np.array([i for i, m in info.items() if m["split"] == "train"])
    te_ids = np.array([i for i, m in info.items() if m["split"] == "test"])
    cc_tr = np.array([i for i in tr_ids if info[i]["view"] == "CC" and i in oof_all])
    mlo_tr = np.array([i for i in tr_ids if info[i]["view"] == "MLO" and i in oof_all])
    cc_te = np.array([i for i in te_ids if info[i]["view"] == "CC" and i in te_all])
    mlo_te = np.array([i for i in te_ids if info[i]["view"] == "MLO" and i in te_all])

    thr_branches = _branches(tr_ids, cc_tr, mlo_tr, oof_all, oof_cc, oof_mlo, mode)
    thr_cf = evaluate.case_fuse(conn, thr_branches, method="mean")
    thr = _best_threshold(thr_cf["y_true"], thr_cf["y_score"])
    cv_auc = roc_auc_score(thr_cf["y_true"], thr_cf["y_score"])

    te_branches = _branches(te_ids, cc_te, mlo_te, te_all, te_cc, te_mlo, mode)
    cf = evaluate.case_fuse(conn, te_branches, method="mean")
    y_te, s_te, pidx = cf["y_true"], cf["y_score"], cf["paired_idx"]
    full = _metrics(y_te, s_te, thr)
    pm = _metrics(y_te[pidx], s_te[pidx], thr)

    print(f"  FineTunedProbe[mammo_clip_b5] mode={mode} C={C} thr={thr:.3f} "
          f"TRAIN-CV-case-AUC={cv_auc:.4f}")
    print(f"    FULL({len(y_te)}):   acc={full['accuracy']:.4f} bal_acc={full['bal_acc']:.4f} "
          f"auc={full['auc']:.4f} sens={full['sensitivity']:.4f} spec={full['specificity']:.4f}")
    print(f"    PAIRED({len(pidx)}): acc={pm['accuracy']:.4f} bal_acc={pm['bal_acc']:.4f} "
          f"auc={pm['auc']:.4f} sens={pm['sensitivity']:.4f} spec={pm['specificity']:.4f}")
    return full, pm


def print_cost():
    if not os.path.exists(LOG_JSON):
        print("  (no cache/finetune_log.json -- run finetune_mammo_clip.py first)")
        return
    with open(LOG_JSON) as f:
        log = json.load(f)
    total_hr = sum(e["wall_clock_sec"] for e in log) / 3600
    peak_gb = max(e["peak_gpu_gb"] for e in log)
    print(f"  fine-tuning cost: {total_hr:.2f} GPU-hours across {len(log)} folds "
          f"(peak {peak_gb:.1f}GB VRAM), needing the full {sum(1 for _ in log)}x "
          f"~985-image labelled train pool each time")
    print("  frozen-encoder cost: one forward pass per image, cached, ~seconds on this GPU, "
          "needs 0 labelled images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fourway", choices=["allonly", "spec2", "fourway"])
    ap.add_argument("--C", type=float, default=0.05)
    args = ap.parse_args()

    print("=== FROZEN (existing, no fine-tuning) ===")
    run_frozen(["mammo_clip_b5"], dataset=DATASET, mode=args.mode, C=args.C)

    print("\n=== FINE-TUNED (this control experiment) ===")
    run(mode=args.mode, C=args.C)

    print("\n=== COST ===")
    print_cost()
