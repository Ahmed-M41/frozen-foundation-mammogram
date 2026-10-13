"""Measure the training COST of the multi-view linear probe (the only trained component).

Reports, for CBIS-DDSM and Mini-DDSM, on this machine (CPU only):
  * wall-clock time to fit the full headline probe (3 heads x 5-fold OOF + test scoring),
  * wall-clock time for the single-model deployment variant (head_full, 3 heads),
  * number of trainable parameters in the probe,
  * number of training images / patients,
  * peak resident-memory increment during the fit.

The frozen feature extraction is a separate, one-time cost (features are cached); this
script times ONLY the trainable part, on cached features already in the database.
Run: python benchmark_mvlp.py
"""
import json
import os
import time

import numpy as np
import psutil

import config, db
import miniddsm_B as M
import multiview_probe as MV
from exp_msfeat_fusion import load_enc, BASE3, MSFEAT

ENC4 = BASE3 + [MSFEAT]
POS = ["MALIGNANT", "CANCER"]
REPEATS = 7
FEAT_DIM = 7680          # 2048 + 1024 + 768 + 3840, L2-normed & concatenated


def n_params_head_cv(n_folds=5, n_heads=3, d=FEAT_DIM):
    # head_cv deploys the AVERAGE of n_folds logistic models per head; each model has
    # d weights + 1 intercept. Three heads (view-agnostic, CC, MLO).
    return n_folds * n_heads * (d + 1)


def n_params_head_full(n_heads=3, d=FEAT_DIM):
    return n_heads * (d + 1)


def time_fit(fn, repeats=REPEATS):
    proc = psutil.Process()
    ts, mem = [], []
    for _ in range(repeats):
        rss0 = proc.memory_info().rss
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        mem.append(max(0, proc.memory_info().rss - rss0))
    ts = np.array(ts)
    # drop the first (warm-up / allocator) run when averaging if we have spares
    core = ts[1:] if len(ts) > 2 else ts
    return dict(mean=float(core.mean()), std=float(core.std()),
                best=float(ts.min()), peak_mem_mb=float(max(mem) / 1e6))


def bench_dataset(tag, ids, X, y, grp, view, tr, te):
    ntr_img, ntr_pat = int(tr.sum()), int(np.unique(grp[tr]).size)
    nte_img = int(te.sum())
    cc, mlo = view == "CC", view == "MLO"
    print(f"\n[{tag}] train={ntr_img} imgs / {ntr_pat} patients   test={nte_img} imgs", flush=True)

    # Isolate the trained component: the three heads, each 5-fold OOF fit + test scoring.
    # This is exactly what "training the probe" costs; no DB / case-fusion I/O in the timing.
    def fit_heads(head_fn):
        def _():
            head_fn(X[tr], y[tr], grp[tr], X[te], M.C_HEAD, seed=0)                     # view-agnostic
            head_fn(X[tr & cc], y[tr & cc], grp[tr & cc], X[te & cc], M.C_HEAD, seed=0)  # CC
            head_fn(X[tr & mlo], y[tr & mlo], grp[tr & mlo], X[te & mlo], M.C_HEAD, seed=0)  # MLO
        return _

    cv = time_fit(fit_heads(MV.head_cv))
    print(f"[{tag}] head_cv  (headline, 3x5 fits): "
          f"mean={cv['mean']*1000:.0f} ms  best={cv['best']*1000:.0f} ms  "
          f"peakRAM+={cv['peak_mem_mb']:.0f} MB", flush=True)

    full = time_fit(fit_heads(MV.head_full))
    print(f"[{tag}] head_full(deploy,   3   fits): "
          f"mean={full['mean']*1000:.0f} ms  best={full['best']*1000:.0f} ms", flush=True)

    return dict(tag=tag, train_imgs=ntr_img, train_patients=ntr_pat, test_imgs=nte_img,
                head_cv=cv, head_full=full,
                params_head_cv=n_params_head_cv(), params_head_full=n_params_head_full())


def main():
    conn = db.connect()
    out = {}

    # CBIS-DDSM benign/malignant
    ids, X, y0, split, grp, view, ds, patho, abn = load_enc(conn, ["cbis_full"], ENC4)
    y = np.isin(patho, POS).astype(int)
    tr = (ds == "cbis_full") & (split == "train")
    te = (ds == "cbis_full") & (split == "test")
    out["cbis"] = bench_dataset("CBIS", ids, X, y, grp, view, tr, te)

    # Mini-DDSM normal/cancer
    raw = load_enc(conn, ["miniddsm", "miniddsm_normal"], ENC4)
    ids, X, y, split, grp, view, ds, patho, abn = M.apply_task("nc", *raw)
    te = M.patient_test_mask(grp, y, 0.2, 0)
    tr = ~te
    out["mini"] = bench_dataset("MINI", ids, X, y, grp, view, tr, te)

    out["meta"] = dict(feat_dim=FEAT_DIM, C=M.C_HEAD, n_folds=5, n_heads=3,
                       repeats=REPEATS)
    path = os.path.join(config.PROJECT_DIR, "cache", "mvlp_cost.json")
    json.dump(out, open(path, "w"), indent=2)
    print("\nsaved", path)
    print("\n=== SUMMARY (trainable parameters) ===")
    print(f"  head_cv deploy params:   {n_params_head_cv():,}")
    print(f"  head_full deploy params: {n_params_head_full():,}")


if __name__ == "__main__":
    main()
