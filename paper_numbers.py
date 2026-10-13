"""Compute the exact Ensembler B numbers used in the IEEE Access paper, all under
ONE protocol: fixed fusion weight w=0.6 (no per-case tuning) + HONEST threshold
picked on train (no oracle). Full metrics + confusion matrix, 4 seeds, mean±std.

Configs:
  CBIS-DDSM benign/malignant (mass, official paired-151 test)
  CBIS-DDSM + MIAS  (MIAS augments the RAC head train pool only; 200%)
  Mini-DDSM normal/cancer (20% patient-disjoint test)
  Mini-DDSM normal/cancer + MIAS (200%)

Prints a block per config and writes cache/paper_numbers.json for the figures.
"""
import json
import os

import numpy as np

import config
import db
import miniddsm_B as M

W, THR = 0.6, "train"


def cbis_ms():
    full = {}
    for sp in ("train", "test"):
        z = np.load(os.path.join(config.PROJECT_DIR, "cache", f"mammoscreen_{sp}.npz"),
                    allow_pickle=True)
        full.update({str(c): float(s) for c, s in zip(z["cids"], z["s"])})
    return full


def run_cfg(conn, datasets, tr_te_fn, ms_map, label):
    ids, X, y, split, grp, view, ds, patho, abn = M.load_base(conn, datasets)
    per = []
    for seed in M.SEEDS:
        tr, te = tr_te_fn(ds, split)
        m, _, _ = M.evaluate_once(conn, ids, X, y, grp, view, tr, te, ms_map, W, THR, seed)
        per.append(m)
    a = M.agg_cm(per)
    M.print_block(label, a, len(M.SEEDS))
    return a


def main():
    conn = db.connect()
    ms_c = cbis_ms()
    out = {}

    # --- CBIS benign/malignant (official split) ---
    out["cbis_bm"] = run_cfg(
        conn, ["cbis_full"],
        lambda ds, sp: ((ds == "cbis_full") & (sp == "train"),
                        (ds == "cbis_full") & (sp == "test")),
        ms_c, "CBIS-DDSM  benign/malignant (mass)  w=0.6 honest")

    # --- CBIS + MIAS 200% (MIAS -> head train pool only) ---
    out["cbis_mias200"] = run_cfg(
        conn, ["cbis_full", "mias_full"],
        lambda ds, sp: (((ds == "cbis_full") & (sp == "train")) | (ds == "mias_full"),
                        (ds == "cbis_full") & (sp == "test")),
        ms_c, "CBIS-DDSM + MIAS (200%)  w=0.6 honest")

    # --- Mini-DDSM normal/cancer (runtime 20% split, seed-varying) ---
    dm = M.apply_task("nc", *M.load_base(conn, ["miniddsm", "miniddsm_normal"]))
    ms_m = M.load_ms(["miniddsm", "miniddsm_normal"])
    ids, X, y, _, grp, view, ds, patho, abn = dm
    per = []
    for seed in M.SEEDS:
        te = M.patient_test_mask(grp, y, 0.2, seed)
        m, _, _ = M.evaluate_once(conn, ids, X, y, grp, view, ~te, te, ms_m, W, THR, seed)
        per.append(m)
    out["miniddsm_nc"] = M.agg_cm(per)
    M.print_block("Mini-DDSM  normal/cancer  w=0.6 honest", out["miniddsm_nc"], len(M.SEEDS))

    # persist (drop the nested 'std' dict into flat keys for the figures)
    with open(os.path.join(config.PROJECT_DIR, "cache", "paper_numbers.json"), "w") as f:
        json.dump({k: {kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in v.items()}
                   for k, v in out.items()}, f, indent=2)
    print("\nsaved cache/paper_numbers.json")


if __name__ == "__main__":
    main()
