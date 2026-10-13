"""Ensembler A (feat-head + RAC-head + mammoscreen-score) version of the extend
experiment — same protocol as data_scaling_extend.py (which does Ensembler B), so
A and B can be compared under adding MIAS/InBreast/CMMD to the database.

A adds a third branch: a multi-view head on mammoscreen_feat (3840-dim) features.
The extra dataset augments BOTH the feat-head and RAC-head training pools; the
frozen mammoscreen SCORE and all calibration stay on CBIS. Fusion = 3-way
percentile, weights tuned on CBIS-train AUC (grid).

Run:  python data_scaling_extend_A.py --extra cmmd_full --tag cmmd --label "CMMD (digital)"
Outputs: output/data_scaling_A_{tag}_{auc,acc,bal}.png, output/data_scaling_A_{tag}.csv
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import config
import db
from mil import _metrics  # noqa: F401  (kept for parity/debug)
from data_scaling import load_ms_maps, _pct, align, ensembler_A
import data_scaling_extend as EXT

FEAT = ["mammoscreen_feat"]
BASE = EXT.BASE
C_FEAT, C_BASE = 0.2, 0.5
CBIS_DS = "cbis_full"
POINTS = EXT.POINTS
SEEDS = EXT.SEEDS


def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def load_both(conn, extra):
    """Load mammoscreen_feat and BASE features for [cbis_full, extra], aligned."""
    def load_set(encs):
        ids0, mats = None, []
        for e in encs:
            ids, v, _ = db.load_embeddings(conn, e, dataset=[CBIS_DS, extra])
            if ids0 is None:
                ids0 = ids
            else:
                assert np.array_equal(ids, ids0), f"misalign {e}"
            mats.append(_l2(v))
        return ids0, np.concatenate(mats, 1).astype(np.float32)

    idf, Xf = load_set(FEAT)
    idb, Xb = load_set(BASE)
    assert np.array_equal(idf, idb), "feat/BASE sample ids misaligned"
    meta = db.sample_meta(conn, idf)
    info = {r["id"]: r for r in conn.execute(
        "SELECT id,split FROM samples WHERE dataset IN (?,?)", (CBIS_DS, extra))}
    split = np.array([info[int(i)]["split"] for i in idf])
    y = np.array([meta[int(i)]["binary_label"] for i in idf])
    view = np.array([meta[int(i)]["view"] for i in idf])
    ds = np.array([meta[int(i)]["dataset"] for i in idf])
    grp = np.array([meta[int(i)]["case_id"] if meta[int(i)]["dataset"] == extra
                    else meta[int(i)]["patient_id"] for i in idf])
    keep = np.isin(view, ["CC", "MLO"])
    return (np.asarray(idf)[keep], Xf[keep], Xb[keep], y[keep], split[keep],
            grp[keep], view[keep], ds[keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="cmmd_full")
    ap.add_argument("--tag", default="cmmd")
    ap.add_argument("--label", default="CMMD (digital)")
    args = ap.parse_args()
    EXT.IB_DS = args.extra            # so EXT.build_train_mask targets the right extra ds
    extra, tag, label = args.extra, args.tag, args.label

    conn = db.connect()
    ids, Xf, Xb, y, split, grp, view, ds = load_both(conn, extra)
    ms_tr, ms_te = load_ms_maps()
    te = split == "test"

    rows = []
    for pt in POINTS:
        cf, gf = EXT.point_to_fracs(pt)
        for seed in SEEDS:
            if pt == 0:
                if seed != SEEDS[0]:
                    continue
                trm = EXT.build_train_mask(y, split, ds, grp, 1.0, 0.0, 0)
                _, rcf_te = EXT.head_case_scores(conn, ids, Xb, y, grp, view, trm, te, C_BASE, 0)
                cids = np.array([str(c) for c in rcf_te["cids"]])
                pidx = np.array(rcf_te["paired_idx"], dtype=int)
                keep = np.array([i for i in pidx if cids[i] in ms_te], dtype=int)
                yte_p = np.array(rcf_te["y_true"])[keep]
                Mte_p = np.array([ms_te[cids[i]] for i in keep])
                auc = roc_auc_score(yte_p, Mte_p)
                mm = _metrics(yte_p, Mte_p, 0.5)
                for s in SEEDS:
                    rows.append(dict(pt=pt, seed=s, n_cbis=0, n_ib=0, n_img=0,
                                     auc=auc, acc=mm["accuracy"], bal=mm["bal_acc"]))
                print(f"pt={pt:3d} seed=all  DB=0            | A(mammoscreen only) "
                      f"auc={auc:.4f} bal={mm['bal_acc']:.4f}", flush=True)
                continue

            trm = EXT.build_train_mask(y, split, ds, grp, cf, gf, seed)
            n_cbis = int((trm & (ds == CBIS_DS)).sum())
            n_ib = int((trm & (ds == extra)).sum())
            fcf_tr, fcf_te = EXT.head_case_scores(conn, ids, Xf, y, grp, view, trm, te, C_FEAT, seed)
            rcf_tr, rcf_te = EXT.head_case_scores(conn, ids, Xb, y, grp, view, trm, te, C_BASE, seed)
            ytr, Ftr, Rtr, Mtr = align(fcf_tr, rcf_tr, ms_tr)
            yte, Fte, Rte, Mte = align(fcf_te, rcf_te, ms_te)
            ctr = [_pct(Ftr, Ftr), _pct(Rtr, Rtr), _pct(Mtr, Mtr)]
            cte = [_pct(Fte, Ftr), _pct(Rte, Rtr), _pct(Mte, Mtr)]
            auc, acc, bal = ensembler_A(ytr, ctr, yte, cte)
            rows.append(dict(pt=pt, seed=seed, n_cbis=n_cbis, n_ib=n_ib,
                             n_img=n_cbis + n_ib, auc=auc, acc=acc, bal=bal))
            print(f"pt={pt:3d} seed={seed} DB={n_cbis:4d}cbis+{n_ib:4d}ex | "
                  f"A auc={auc:.4f} acc={acc:.4f} bal={bal:.4f}", flush=True)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(config.OUTPUT_DIR, f"data_scaling_A_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    pts = np.array(POINTS)
    agg = {}
    for metric in ["auc", "acc", "bal"]:
        mean = np.array([np.mean([r[metric] for r in rows if r["pt"] == p]) for p in POINTS])
        std = np.array([np.std([r[metric] for r in rows if r["pt"] == p]) for p in POINTS])
        agg[metric] = (mean, std)
    nimg = np.array([np.mean([r["n_img"] for r in rows if r["pt"] == p]) for p in POINTS])

    titles = {"auc": "Case-level AUC", "acc": "Accuracy", "bal": "Balanced accuracy"}
    for metric in ["auc", "acc", "bal"]:
        mean, std = agg[metric]
        plt.figure(figsize=(9, 5))
        plt.plot(pts, mean, "-o", color="#1f77b4", label="Ensembler A (feat + RAC + score)")
        plt.fill_between(pts, mean - std, mean + std, color="#1f77b4", alpha=0.15)
        plt.axvline(100, color="gray", ls="--", lw=1)
        plt.text(101, mean.min(), f"  + {label} added -->", color="gray", fontsize=9, va="bottom")
        plt.xlabel(f"% of database used   (0-100 = CBIS-train;  100-200 = full CBIS + % of {label})")
        plt.ylabel(titles[metric])
        plt.title(f"[Ensembler A] {titles[metric]} vs. database size, extended with {label} "
                  f"(paired-151, {len(SEEDS)} seeds)")
        plt.xticks(pts)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        out = os.path.join(config.OUTPUT_DIR, f"data_scaling_A_{tag}_{metric}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=140)
        plt.close()
        print("saved", out)

    ma, _ = agg["auc"]
    i100 = POINTS.index(100)
    print(f"\n=== Ensembler A: does adding {label} beyond 100% help? (AUC) ===")
    print(f"  0%   (mammoscreen only) = {ma[0]:.4f}")
    print(f"  100% (full CBIS)        = {ma[i100]:.4f}  [db~{int(nimg[i100])} imgs]")
    print(f"  200% (CBIS + all extra) = {ma[-1]:.4f}  [db~{int(nimg[-1])} imgs]")
    print(f"  delta(200-100)          = {ma[-1]-ma[i100]:+.4f}")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
