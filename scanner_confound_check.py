"""Reviewer diagnostic: does the Mini-DDSM normal-vs-cancer AUC (0.982 in the paper)
survive when the acquisition/scanner shortcut is removed?

Mini-DDSM's case-name letter prefix (A/B/C/D) tracks the original DDSM
institution+scanner (Heath et al.): A = MGH (DBA M2100 ImageClear), B/C = WFU
(Lumisys 200 Laser), D = WUSM (Howtek MultiRad850). In the normal-vs-cancer subset
used here, prefixes B and C contribute ZERO normal cases (620+932=1552 cancer
images, 0 normal), while prefix A alone supplies 83% of all normal images. So
scanner identity is confounded with the label almost by construction across the
full dataset.

Two diagnostics, both reusing the paper's exact cached BASE3 features (Mammo-CLIP +
DINOv2 + RAD-DINO) and classifier code (no re-extraction, no GPU):

1. scanner-predictability: can the same frozen features + linear head predict
   scanner-prefix (A vs D, the two prefixes that both contain normal cases) well
   above chance? High accuracy would mean the representation trivially encodes
   acquisition signature.
2. single-scanner control: restrict BOTH train and test to prefix A only (2000
   normal / 768 cancer images -- the one prefix with enough of both classes to run
   the task on its own). Train and test the identical 3-encoder classifier with no
   possible between-scanner shortcut, patient-grouped, 4 seeds. Compare its AUC to
   the all-prefix 0.977 (Table tab:components) headline.

Run: python scanner_confound_check.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import config
import db
import evaluate
import miniddsm_B as M
from data_scaling_extend import head_case_scores
from mil import _best_threshold, _metrics

SEEDS = [0, 1, 2, 3]
C_HEAD = 0.5


def _prefix(grp):
    return np.array([g.split("_")[0] for g in grp])


def diag_scanner_predict(ids, X, grp, view, prefix, y_nc):
    """Can BASE3 features tell prefix A from prefix D (both real scanners, both
    containing genuine normal AND cancer cases)?

    FIX (2026-11-12): the submitted version scored this AUC against the cancer/normal
    label (te_cf["y_true"] from evaluate.case_fuse reads binary_label from the DB),
    not against the site label it was trained on. Now: image-level, patient-grouped
    linear probe trained AND scored on site labels. Also run within each class, so the
    cancer/normal label cannot drive the site prediction."""
    import multiview_probe as MV
    res = {}
    for name, cls in (("all", None), ("normal only", 0), ("cancer only", 1)):
        keep = np.isin(prefix, ["A", "D"]) & (True if cls is None else (y_nc == cls))
        Xs, gs = X[keep], grp[keep]
        ysite = (prefix[keep] == "A").astype(int)    # 1 = prefix A, 0 = prefix D
        aucs = []
        for seed in SEEDS:
            te = M.patient_test_mask(gs, ysite, 0.2, seed)
            _, s_te = MV.head_cv(Xs[~te], ysite[~te], gs[~te], Xs[te], C_HEAD, seed=seed)
            aucs.append(roc_auc_score(ysite[te], s_te))
        res[name] = (float(np.mean(aucs)), float(np.std(aucs)), int(keep.sum()))
        print(f"  site A vs D from BASE3 features [{name:11s}]: AUC={np.mean(aucs):.4f} "
              f"+- {np.std(aucs):.4f} over {len(SEEDS)} seeds (n={int(keep.sum())} images; "
              f"A={int(ysite.sum())}, D={int((1 - ysite).sum())})")
    return res


def diag_single_scanner_nc(ids, X, y, grp, view, prefix):
    """Normal-vs-cancer classifier trained AND tested entirely within prefix A:
    no between-scanner shortcut is available at all."""
    keep = prefix == "A"
    ids, X, y, grp, view = ids[keep], X[keep], y[keep], grp[keep], view[keep]
    n_breasts_pos = len(np.unique(grp[y == 1]))
    n_breasts_neg = len(np.unique(grp[y == 0]))
    print(f"  prefix-A-only subset: {len(ids)} images, "
          f"{n_breasts_pos} cancer patients, {n_breasts_neg} normal patients")

    aucs, accs, bals = [], [], []
    conn = db.connect()
    for seed in SEEDS:
        te_mask = M.patient_test_mask(grp, y, 0.2, seed)
        tr_mask = ~te_mask
        tr_cf, te_cf = head_case_scores(conn, ids, X, y, grp, view, tr_mask, te_mask,
                                         C_HEAD, seed)
        thr = config.DECISION_THRESHOLD   # fixed 0.5, untuned (revision 2026-11-13)
        m = _metrics(te_cf["y_true"], te_cf["y_score"], thr)
        aucs.append(roc_auc_score(te_cf["y_true"], te_cf["y_score"]))
        accs.append(m["accuracy"]); bals.append(m["bal_acc"])
    print(f"  single-scanner (prefix A only) normal-vs-cancer: "
          f"AUC={np.mean(aucs):.4f} +- {np.std(aucs):.4f}  "
          f"acc={np.mean(accs)*100:.1f}%  bal_acc={np.mean(bals)*100:.1f}% "
          f"over {len(SEEDS)} seeds")
    return np.mean(aucs)


def main():
    conn = db.connect()
    raw = M.load_base(conn, ["miniddsm", "miniddsm_normal"])
    ids, X, y, split, grp, view, ds, patho, abn = M.apply_task("nc", *raw)
    prefix = _prefix(grp)

    print("=== prefix breakdown of the evaluated normal-vs-cancer image set ===")
    for p in sorted(set(prefix)):
        m = prefix == p
        print(f"  {p}: cancer={int((y[m] == 1).sum()):4d}  normal={int((y[m] == 0).sum()):4d}")

    print("\n=== diagnostic 1: scanner predictability (prefix A vs D) ===")
    diag_scanner_predict(ids, X, grp, view, prefix, y)

    print("\n=== diagnostic 2: single-scanner (prefix A only) normal-vs-cancer ===")
    diag_single_scanner_nc(ids, X, y, grp, view, prefix)

    print("\n(compare against the all-prefix 3-encoder classifier headline: 0.977 AUC,"
          " Table tab:components)")


if __name__ == "__main__":
    main()
