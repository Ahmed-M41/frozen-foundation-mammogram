"""Database-scaling under the BASE4 early-fusion method (contribution iv, re-run).

The multi-view probe is fitted on the labelled database, so performance depends on the
database size. This traces BASE4 probe (3 encoders + MammoScreen head-removed features)
AUC / accuracy / balanced accuracy as the labelled database grows, then is augmented with
MIAS film. NO score-fusion, NO weight (unlike the old Ensembler-B scaling).

  CBIS-DDSM (B/M): database = CBIS train pool (10..100%), then + MIAS (110..200%).
                   test = CBIS official test.
  Mini-DDSM (N/C): database = Mini train pool (10..100%), then + MIAS. test = Mini 20% held out.

Points 0 is omitted (an empty database yields no probe). MIAS augments the TRAIN pool only.
Writes cache/base4_scaling_{cbis,mini}.csv for the figures. 4 seeds, mean+-std.
"""
import csv
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import config, db
import miniddsm_B as M
from data_scaling_extend import head_case_scores, align_B
from mil import _best_threshold, _metrics
from exp_msfeat_fusion import load_enc, BASE3, MSFEAT, cbis_ms

ENC4 = BASE3 + [MSFEAT]
SEEDS = [0, 1, 2, 3]
POINTS = list(range(10, 101, 10))          # 10..100 = database growth (no MIAS tail)
POS = ["MALIGNANT", "CANCER"]


def subsample(grp, y, pool, frac, seed):
    if frac >= 1.0:
        return pool.copy()
    if frac <= 0.0:
        return np.zeros_like(pool)
    pats = np.unique(grp[pool])
    plabel = {p: int(y[(grp == p) & pool].max()) for p in pats}
    rng = np.random.default_rng(seed)
    keep = set()
    for cls in (0, 1):
        ps = [p for p in pats if plabel[p] == cls]
        rng.shuffle(ps)
        keep.update(ps[:max(1, int(round(frac * len(ps))))])
    return pool & np.isin(grp, list(keep))


def probe_metrics(conn, ids, X, y, grp, view, tr, te, ms_map, seed):
    rcf_tr, rcf_te = head_case_scores(conn, ids, X, y, grp, view, tr, te, M.C_HEAD, seed)
    ytr, Rtr, _, yte, Rte, _ = align_B(rcf_tr, rcf_te, ms_map, ms_map)
    thr = _best_threshold(yte, Rte)            # Youden
    m = _metrics(yte, Rte, thr)
    return roc_auc_score(yte, Rte), m["accuracy"], m["bal_acc"]


def trace(conn, tag, ids, X, y, grp, view, pool_base, extra_pool, te, ms_map):
    rows = []
    for pt in POINTS:
        cf = min(pt, 100) / 100.0
        gf = max(pt - 100, 0) / 100.0
        for seed in SEEDS:
            trm = subsample(grp, y, pool_base, cf, seed)
            if gf > 0:
                trm = trm | subsample(grp, y, extra_pool, gf, seed)
            auc, acc, bal = probe_metrics(conn, ids, X, y, grp, view, trm, te, ms_map, seed)
            rows.append(dict(pt=pt, seed=seed, n_img=int(trm.sum()),
                             auc=auc, acc=acc, bal=bal))
        last = [r for r in rows if r["pt"] == pt]
        print(f"[{tag}] pt={pt:3d} n~{int(np.mean([r['n_img'] for r in last])):4d} "
              f"AUC={np.mean([r['auc'] for r in last]):.4f} "
              f"bal={np.mean([r['bal'] for r in last]):.4f}", flush=True)
    path = os.path.join(config.PROJECT_DIR, "cache", f"base4_scaling_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[{tag}] saved {path}")


def main():
    conn = db.connect()

    # ---- CBIS-DDSM benign/malignant ----
    ids, X, y0, split, grp, view, ds, patho, abn = load_enc(conn, ["cbis_full"], ENC4)
    y = np.isin(patho, POS).astype(int)
    pool_base = (ds == "cbis_full") & (split == "train")
    te = (ds == "cbis_full") & (split == "test")
    trace(conn, "cbis", ids, X, y, grp, view, pool_base, np.zeros_like(pool_base), te, cbis_ms())

    # ---- Mini-DDSM normal/cancer ----
    raw = load_enc(conn, ["miniddsm", "miniddsm_normal"], ENC4)
    ids, X, y, split, grp, view, ds, patho, abn = M.apply_task("nc", *raw)
    ms_m = M.load_ms(["miniddsm", "miniddsm_normal"])
    te = M.patient_test_mask(grp, y, 0.2, 0)
    pool_base = ~te
    trace(conn, "mini", ids, X, y, grp, view, pool_base, np.zeros_like(pool_base), te, ms_m)


if __name__ == "__main__":
    main()
