"""Ablation (user-requested): is the percentile-rank normalisation step in the
fusion doing real work, or can it be dropped?

For CBIS-DDSM benign/malignant (fusion HELPS) and Mini-DDSM normal/cancer (fusion
does not), compare three ways of combining R (probe) and M (MammoScreen) at the
same fixed w=0.6, honest train-fit normalisation:
  percentile : S = w*pct(R) + (1-w)*pct(M)          [current method]
  raw        : S = w*R      + (1-w)*M               [no normalisation]
  z-score    : S = w*z(R)   + (1-w)*z(M)            [standardise on train]
Reports test AUC (mean over the paper's 4 seeds) plus R-alone / M-alone.
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import config, db
import miniddsm_B as M
from data_scaling_extend import head_case_scores, align_B
from data_scaling import _pct

W = 0.6


def cbis_ms():
    full = {}
    for sp in ("train", "test"):
        z = np.load(os.path.join(config.PROJECT_DIR, "cache", f"mammoscreen_{sp}.npz"),
                    allow_pickle=True)
        full.update({str(c): float(s) for c, s in zip(z["cids"], z["s"])})
    return full


def _z(x, ref):
    mu, sd = np.mean(ref), np.std(ref) + 1e-9
    return (x - mu) / sd


def fusions(ytr, Rtr, Mtr, yte, Rte, Mte):
    # percentile
    fp = W * _pct(Rte, Rtr) + (1 - W) * _pct(Mte, Mtr)
    # raw
    fr = W * Rte + (1 - W) * Mte
    # z-score
    fz = W * _z(Rte, Rtr) + (1 - W) * _z(Mte, Mtr)
    return {"percentile": roc_auc_score(yte, fp),
            "raw": roc_auc_score(yte, fr),
            "z-score": roc_auc_score(yte, fz),
            "R_alone": roc_auc_score(yte, Rte),
            "M_alone": roc_auc_score(yte, Mte)}


def run(conn, datasets, tr_te_fn, ms_map, label, apply_nc=False):
    if apply_nc:
        ids, X, y, split, grp, view, ds, patho, abn = M.apply_task(
            "nc", *M.load_base(conn, datasets))
    else:
        ids, X, y, split, grp, view, ds, patho, abn = M.load_base(conn, datasets)
    acc = {}
    for seed in M.SEEDS:
        tr, te = tr_te_fn(ds, split, grp, y, seed)
        rcf_tr, rcf_te = head_case_scores(conn, ids, X, y, grp, view, tr, te, M.C_HEAD, seed)
        ytr, Rtr, Mtr, yte, Rte, Mte = align_B(rcf_tr, rcf_te, ms_map, ms_map)
        for k, v in fusions(ytr, Rtr, Mtr, yte, Rte, Mte).items():
            acc.setdefault(k, []).append(v)
    print(f"\n[{label}]  (mean over {len(M.SEEDS)} seeds)")
    for k in ("R_alone", "M_alone", "percentile", "raw", "z-score"):
        print(f"    {k:11s} AUC = {np.mean(acc[k]):.3f} ± {np.std(acc[k]):.3f}")
    return acc


def main():
    conn = db.connect()
    run(conn, ["cbis_full"],
        lambda ds, sp, grp, y, seed: ((ds == "cbis_full") & (sp == "train"),
                                      (ds == "cbis_full") & (sp == "test")),
        cbis_ms(), "CBIS-DDSM benign/malignant (fusion helps here)")
    run(conn, ["miniddsm", "miniddsm_normal"],
        lambda ds, sp, grp, y, seed: (~M.patient_test_mask(grp, y, 0.2, seed),
                                      M.patient_test_mask(grp, y, 0.2, seed)),
        M.load_ms(["miniddsm", "miniddsm_normal"]),
        "Mini-DDSM normal/cancer (fusion does not help)", apply_nc=True)


if __name__ == "__main__":
    main()
