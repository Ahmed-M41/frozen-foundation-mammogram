"""Database-scaling under the BASE4 early-fusion method (the paper's actual
architecture: single multi-view probe on 4 concatenated extractors, NO late
score-fusion), extended past 100% with an out-of-domain digital dataset
(InBreast / CMMD), to re-test "does more data help" under the SAME method
that produced the paper's headline numbers (0.904 CBIS / 0.982 Mini-DDSM).

This replaces the old Ensembler-B extend experiment (data_scaling_extend.py),
which measured the out-of-domain-hurts finding under the weaker LATE-fusion
variant (RAC head + percentile-fused mammoscreen score) -- architecturally the
same as this paper's own "Late fusion" row in Table 6, which the paper's own
ablation already shows is inferior to the adopted early-fusion method.

x-axis: 10,20,...,100 (CBIS-train fraction, film) then 110,...,200
(full CBIS train + fraction of the extra out-of-domain set). Point 0 omitted
(an empty database yields no probe -- see base4_scaling.py).

Run:  python base4_scaling_extend.py --extra inbreast_full --tag ib --label "InBreast (digital)"
      python base4_scaling_extend.py --extra cmmd_full --tag cmmd --label "CMMD (digital)"
"""
import argparse
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import config
import db
from data_scaling_extend import head_case_scores, align_B, _l2
from exp_msfeat_fusion import BASE3, MSFEAT, cbis_ms
from miniddsm_B import C_HEAD
from mil import _best_threshold, _metrics

ENC4 = BASE3 + [MSFEAT]
CBIS_DS = "cbis_full"
POINTS = list(range(10, 201, 10))          # 10..200, 0 omitted (no probe w/o data)
SEEDS = [0, 1, 2, 3]


# ---------------------------------------------------------------- data loading
def load_base4_combined(conn, extra_ds):
    """BASE4 features (3 native encoders + mammoscreen_feat) for CBIS + extra,
    aligned by sample id. Mirrors data_scaling_extend.load_base_combined but
    with the 4-extractor concatenation instead of 3."""
    ids0, mats = None, []
    for e in ENC4:
        ids, v, _ = db.load_embeddings(conn, e, dataset=[CBIS_DS, extra_ds])
        if ids0 is None:
            ids0 = ids
        else:
            assert np.array_equal(ids, ids0), f"misalign {e}"
        mats.append(_l2(v))
    X = np.concatenate(mats, 1).astype(np.float32)
    meta = db.sample_meta(conn, ids0)
    info = {r["id"]: r for r in conn.execute(
        "SELECT id,split FROM samples WHERE dataset IN (?,?)", (CBIS_DS, extra_ds))}
    split = np.array([info[int(i)]["split"] for i in ids0])
    y = np.array([meta[int(i)]["binary_label"] for i in ids0])
    view = np.array([meta[int(i)]["view"] for i in ids0])
    ds = np.array([meta[int(i)]["dataset"] for i in ids0])
    grp = np.array([meta[int(i)]["case_id"] if meta[int(i)]["dataset"] == extra_ds
                    else meta[int(i)]["patient_id"] for i in ids0])
    keep = np.isin(view, ["CC", "MLO"])
    return (np.asarray(ids0)[keep], X[keep], y[keep], split[keep],
            grp[keep], view[keep], ds[keep])


# ---------------------------------------------------------------- subsampling
def build_train_mask(y, split, ds, grp, cbis_frac, extra_frac, seed, extra_ds):
    """Patient-stratified CBIS subsample (film) OR whole-image extra-set subsample
    (digital), unioned into one training mask. Test stays untouched."""
    tr = split == "train"
    cbis_tr = tr & (ds == CBIS_DS)
    extra_tr = tr & (ds == extra_ds)
    rng = np.random.default_rng(seed)

    def pick(mask, frac):
        if frac >= 1.0:
            return mask.copy()
        if frac <= 0.0:
            return np.zeros_like(mask)
        groups = grp[mask]
        uniq = np.unique(groups)
        glabel = {g: int(y[mask & (grp == g)].max()) for g in uniq}
        keep = set()
        for cls in (0, 1):
            gs = [g for g in uniq if glabel[g] == cls]
            rng.shuffle(gs)
            n = max(1, int(round(frac * len(gs))))
            keep.update(gs[:n])
        return mask & np.isin(grp, list(keep))

    return pick(cbis_tr, cbis_frac) | pick(extra_tr, extra_frac)


def point_to_fracs(pt):
    if pt <= 100:
        return pt / 100.0, 0.0
    return 1.0, (pt - 100) / 100.0


# ---------------------------------------------------------------- BASE4 probe
def probe_metrics(conn, ids, X, y, grp, view, tr, te, ms_map, seed):
    rcf_tr, rcf_te = head_case_scores(conn, ids, X, y, grp, view, tr, te, C_HEAD, seed)
    _, _, _, yte, Rte, _ = align_B(rcf_tr, rcf_te, ms_map, ms_map)
    thr = _best_threshold(yte, Rte)            # Youden, same convention as the paper
    m = _metrics(yte, Rte, thr)
    return roc_auc_score(yte, Rte), m["accuracy"], m["bal_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", required=True, help="extra out-of-domain dataset name in the DB")
    ap.add_argument("--tag", required=True, help="output filename tag")
    ap.add_argument("--label", required=True, help="plot/print label for the extra set")
    ap.add_argument("--outdir", default=config.OUTPUT_DIR)
    args = ap.parse_args()
    extra_ds, tag, label = args.extra, args.tag, args.label

    conn = db.connect()
    ids, X, y, split, grp, view, ds = load_base4_combined(conn, extra_ds)
    ms_map = cbis_ms()
    te = (ds == CBIS_DS) & (split == "test")

    rows = []
    for pt in POINTS:
        cf, gf = point_to_fracs(pt)
        for seed in SEEDS:
            trm = build_train_mask(y, split, ds, grp, cf, gf, seed, extra_ds)
            n_cbis = int((trm & (ds == CBIS_DS)).sum())
            n_extra = int((trm & (ds == extra_ds)).sum())
            auc, acc, bal = probe_metrics(conn, ids, X, y, grp, view, trm, te, ms_map, seed)
            rows.append(dict(pt=pt, seed=seed, n_cbis=n_cbis, n_extra=n_extra,
                             n_img=n_cbis + n_extra, auc=auc, acc=acc, bal=bal))
            print(f"pt={pt:3d} seed={seed} DB={n_cbis:4d}cbis+{n_extra:4d}{tag} | "
                  f"BASE4 auc={auc:.4f} acc={acc:.4f} bal={bal:.4f}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"base4_scaling_extend_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pts = np.array(POINTS)
    ma = np.array([np.mean([r["auc"] for r in rows if r["pt"] == p]) for p in POINTS])
    mb = np.array([np.mean([r["bal"] for r in rows if r["pt"] == p]) for p in POINTS])
    i100 = POINTS.index(100)
    print(f"\n=== BASE4 (early fusion, the paper's method): does adding {label} beyond 100% help? ===")
    print(f"  100% (full CBIS)         AUC={ma[i100]:.4f}  bal={mb[i100]:.4f}")
    print(f"  200% (CBIS + all extra)  AUC={ma[-1]:.4f}  bal={mb[-1]:.4f}")
    print(f"  delta(200-100)           AUC={ma[-1]-ma[i100]:+.4f}  bal={mb[-1]-mb[i100]:+.4f}")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
