"""Phase 2 of the reviewer-response control: all three native encoders (Mammo-CLIP,
DINOv2 ViT-L/14, RAD-DINO) fine-tuned end to end, concatenated with the still-frozen
MammoScreen (already pre-trained on CBIS-DDSM itself, so not fine-tuned here), read
through the same four-extractor multi-view classifier used for the 0.904 headline.

Same fold-matched, leak-free discipline as finetune_eval_full4.py, extended to three
fine-tuned encoders sharing the SAME fold assignment (cache/finetune_fold_assignment.json):
fold V's classifier is fit only on fold-V-excluded rows, using each encoder's
fold-V backbone (the one that never saw fold V's labels during its own fine-tuning).

The classifier is refit from scratch per fold on the new concatenated features, same
as finetune_eval_full4.py -- never reusing old classifier weights.

Run: python finetune_eval_phase2.py
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
import db
from mil import _best_threshold, _metrics
from multiview_probe import _l2, _fit_tf, _apply_tf, run as run_frozen
import evaluate

CACHE = os.path.join(config.PROJECT_DIR, "cache")
FOLD_JSON = os.path.join(CACHE, "finetune_fold_assignment.json")
DATASET = "cbis_full"
N_SPLITS = 5
FT_PREFIXES = ["mammo_clip_b5_ft", "dinov2_vitl14_r518_ft", "rad_dino_ft"]
FROZEN_ENCODERS = ["mammoscreen_feat"]


def _load_fold_of():
    with open(FOLD_JSON) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _meta(conn):
    rows = conn.execute(
        "SELECT id,split,binary_label,patient_id,view FROM samples WHERE dataset=?",
        (DATASET,)).fetchall()
    return {r["id"]: r for r in rows}


def _frozen_feats(conn):
    ids0, mats = None, []
    for e in FROZEN_ENCODERS:
        ids, v, _ = db.load_embeddings(conn, e, dataset=DATASET)
        if ids0 is None:
            ids0 = ids
        else:
            assert np.array_equal(ids, ids0), f"misalign {e}"
        mats.append(_l2(v))
    X = np.concatenate(mats, axis=1).astype(np.float32)
    return {int(i): X[k] for k, i in enumerate(ids0)}


def head_phase2(conn, fold_of, info, frozen, C, view_filter=None):
    oof, test_scores = {}, {}
    for V in range(N_SPLITS):
        ft_mats, ids0 = [], None
        for prefix in FT_PREFIXES:
            name = f"{prefix}_fold{V}"
            ids, X, _ = db.load_embeddings(conn, name, dataset=DATASET)
            if ids0 is None:
                ids0 = ids
            else:
                assert np.array_equal(ids, ids0), f"misalign {name}"
            ft_mats.append(_l2(X))
        X = np.concatenate(ft_mats + [np.stack([frozen[int(i)] for i in ids0])], axis=1)
        ids = ids0

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


def _branches(ids_subset, cc_ids, mlo_ids, all_map, cc_map, mlo_map):
    return [
        dict(ids=cc_ids, score_pos=np.array([all_map[int(i)] for i in cc_ids])),
        dict(ids=mlo_ids, score_pos=np.array([all_map[int(i)] for i in mlo_ids])),
        dict(ids=cc_ids, score_pos=np.array([cc_map[int(i)] for i in cc_ids])),
        dict(ids=mlo_ids, score_pos=np.array([mlo_map[int(i)] for i in mlo_ids])),
    ]


def run(C=0.05):
    conn = db.connect()
    fold_of = _load_fold_of()
    info = _meta(conn)
    frozen = _frozen_feats(conn)

    oof_all, te_all = head_phase2(conn, fold_of, info, frozen, C, view_filter=None)
    oof_cc, te_cc = head_phase2(conn, fold_of, info, frozen, C, view_filter="CC")
    oof_mlo, te_mlo = head_phase2(conn, fold_of, info, frozen, C, view_filter="MLO")

    tr_ids = np.array([i for i, m in info.items() if m["split"] == "train"])
    te_ids = np.array([i for i, m in info.items() if m["split"] == "test"])
    cc_tr = np.array([i for i in tr_ids if info[i]["view"] == "CC" and i in oof_all])
    mlo_tr = np.array([i for i in tr_ids if info[i]["view"] == "MLO" and i in oof_all])
    cc_te = np.array([i for i in te_ids if info[i]["view"] == "CC" and i in te_all])
    mlo_te = np.array([i for i in te_ids if info[i]["view"] == "MLO" and i in te_all])

    thr_branches = _branches(tr_ids, cc_tr, mlo_tr, oof_all, oof_cc, oof_mlo)
    thr_cf = evaluate.case_fuse(conn, thr_branches, method="mean")
    thr = _best_threshold(thr_cf["y_true"], thr_cf["y_score"])
    cv_auc = roc_auc_score(thr_cf["y_true"], thr_cf["y_score"])

    te_branches = _branches(te_ids, cc_te, mlo_te, te_all, te_cc, te_mlo)
    cf = evaluate.case_fuse(conn, te_branches, method="mean")
    y_te, s_te, pidx = cf["y_true"], cf["y_score"], cf["paired_idx"]
    full = _metrics(y_te, s_te, thr)
    pm = _metrics(y_te[pidx], s_te[pidx], thr)
    run.last_cf = cf

    print(f"  Phase2 (3 fine-tuned + 1 frozen MammoScreen) C={C} thr={thr:.3f} "
          f"TRAIN-CV-case-AUC={cv_auc:.4f}")
    print(f"    FULL({len(y_te)}):   acc={full['accuracy']:.4f} bal_acc={full['bal_acc']:.4f} "
          f"auc={full['auc']:.4f} sens={full['sensitivity']:.4f} spec={full['specificity']:.4f}")
    print(f"    PAIRED({len(pidx)}): acc={pm['accuracy']:.4f} bal_acc={pm['bal_acc']:.4f} "
          f"auc={pm['auc']:.4f} sens={pm['sensitivity']:.4f} spec={pm['specificity']:.4f}")
    return full, pm


def print_cost():
    total_hr = 0.0
    for log_name in ["finetune_log.json", "finetune_raddino_log.json", "finetune_dinov2_log.json"]:
        p = os.path.join(CACHE, log_name)
        if not os.path.exists(p):
            print(f"  (missing {log_name})")
            continue
        with open(p) as f:
            log = json.load(f)
        hr = sum(e["wall_clock_sec"] for e in log) / 3600
        print(f"  {log_name}: {hr:.2f} GPU-hours across {len(log)} folds")
        total_hr += hr
    print(f"  TOTAL: {total_hr:.2f} GPU-hours across all three fine-tuned encoders")


if __name__ == "__main__":
    print("=== FROZEN (all four, existing headline) ===")
    run_frozen(["mammo_clip_b5", "dinov2_vitl14_r518_noclahe", "rad_dino_noclahe",
                "mammoscreen_feat"], dataset=DATASET, mode="fourway", C=0.05)

    print("\n=== PHASE 2: 3 fine-tuned encoders + 1 frozen (MammoScreen) ===")
    run(C=0.05)

    print("\n=== COST ===")
    print_cost()
