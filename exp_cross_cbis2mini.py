"""User-requested generalisation test: TRAIN on CBIS-DDSM, TEST on Mini-DDSM.

Constraint: CBIS has no normal mammograms, so the only trainable task is
benign-vs-malignant (bm). This is a pure cross-DATASET transfer test on the SAME task:

  train pool = CBIS-DDSM  (mass, benign vs malignant)
  test set   = Mini-DDSM  (mass, benign vs cancer)   -- never seen in training

Both use the same frozen encoders, so the probe trained on CBIS embeddings can be
applied directly to Mini-DDSM embeddings. Compares:
  BASE3 probe  (3 frozen encoders)           -- does the probe generalise across datasets?
  BASE4 probe  (3 encoders + mammoscreen_feat) -- does early fusion help or hurt transfer?

No fine-tuning, no retraining on the target. AUC/acc/bal on the Mini-DDSM test set,
honest threshold picked on the CBIS train OOF scores. 4 seeds (CV fold randomisation).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import db
import miniddsm_B as M
from data_scaling_extend import head_case_scores
from mil import _best_threshold, _metrics
from exp_msfeat_fusion import load_enc, BASE3, MSFEAT

POS = ["MALIGNANT", "CANCER"]
BENIGN = ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]
SEEDS = [0, 1, 2, 3]


def eval_cross(conn, X, y, grp, view, ids, tr, te, seed):
    tr_cf, te_cf = head_case_scores(conn, ids, X, y, grp, view, tr, te, M.C_HEAD, seed)
    yte, Rte = np.array(te_cf["y_true"]), np.array(te_cf["y_score"])
    ytr, Rtr = np.array(tr_cf["y_true"]), np.array(tr_cf["y_score"])
    thr = _best_threshold(ytr, Rtr)                 # honest: threshold from CBIS train
    m = _metrics(yte, Rte, thr)
    m_or = _metrics(yte, Rte, _best_threshold(yte, Rte))   # oracle: threshold on target (upper bound)
    return roc_auc_score(yte, Rte), m["accuracy"], m["bal_acc"], m_or["bal_acc"]


def main():
    conn = db.connect()
    ids, X3, _, split, grp, view, ds, patho, abn = load_enc(conn, ["cbis_full", "miniddsm"], BASE3)
    _, X4, *_ = load_enc(conn, ["cbis_full", "miniddsm"], BASE3 + [MSFEAT])

    # same task for both datasets: benign(0) vs malignant/cancer(1); mass only
    y = np.isin(patho, POS).astype(int)
    keep = (abn == "mass") & np.isin(patho, BENIGN + POS)
    ids, X3, X4, y, grp, view, ds = (a[keep] for a in (ids, X3, X4, y, grp, view, ds))

    tr = ds == "cbis_full"      # train pool = CBIS
    te = ds == "miniddsm"       # test set   = Mini-DDSM
    print(f"train (CBIS mass) images={int(tr.sum())}  "
          f"test (Mini-DDSM mass) images={int(te.sum())}  "
          f"[X3={X3.shape[1]}d X4={X4.shape[1]}d]")

    agg = {"R3": [], "R4": []}
    for seed in SEEDS:
        agg["R3"].append(eval_cross(conn, X3, y, grp, view, ids, tr, te, seed))
        agg["R4"].append(eval_cross(conn, X4, y, grp, view, ids, tr, te, seed))

    def show(key, name):
        a = np.array(agg[key]); mu, sd = a.mean(0), a.std(0)
        print(f"    {name:36s} AUC={mu[0]:.4f}±{sd[0]:.4f}  bal(CBIS-thr)={mu[2]:.4f}  "
              f"bal(target-thr)={mu[3]:.4f}")

    print(f"\n=== CBIS -> Mini-DDSM transfer (benign/malignant, mass, {len(SEEDS)} seeds) ===")
    show("R3", "BASE3 probe (3 encoders)")
    show("R4", "BASE4 probe (+ mammoscreen_feat)")


if __name__ == "__main__":
    main()
