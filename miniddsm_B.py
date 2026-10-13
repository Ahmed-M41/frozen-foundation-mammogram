"""Self-contained Ensembler B on Mini-DDSM (and cross-eval on CBIS).

Reproduces the CBIS Ensembler B protocol INSIDE Mini-DDSM: RAC multi-view
logistic probe (StratifiedGroupKFold OOF) + frozen mammoscreen score, percentile-
rank fused (w=0.6 or swept), honest threshold on train. Reuses the exact head /
align / fusion functions from data_scaling_extend so the pipeline is identical.

The train/test split is computed HERE at runtime from patient ids (not the DB
split column), so 10% / 20% / 5-fold can be varied without re-crop/re-embed.

Tasks (--task):
  bm : benign lesion (0) vs malignant lesion (1)  [datasets: miniddsm]        <- CBIS-comparable
  nc : normal breast (0) vs cancer/malignant (1)  [miniddsm_normal + miniddsm] <- Su-et-al style

Modes (--mode):
  single : full train pool as database, held-out test (--test-frac); 4 seeds.
  curve  : database learning curve 0..100% (subsample train pool); single-axes plot.
  extend : 0..200% — 100..200 adds --extra (mias_full) to the RAC head train pool
           only (never in test/calibration); the B_mias-style curve, Mini-DDSM base.
  cv     : k-fold CV (--kfold, patient-grouped) matching the paper's protocol.
  cross  : train DB = Mini-DDSM (task bm), test = CBIS test  [LEAKY — exploratory].

Every terminal config prints AUC / accuracy / balanced acc / sensitivity /
specificity and a confusion matrix (summed over seeds/folds).

Requires (run first): prep_miniddsm.py -> embed_all.py --dataset miniddsm[_normal]
for each BASE encoder -> mammoscreen_cache_miniddsm.py.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import config
import db
from data_scaling import _pct
from data_scaling_extend import BASE, _l2, head_case_scores, align_B
from mil import _best_threshold, _metrics

SEEDS = [0, 1, 2, 3]
C_HEAD = 0.5
POINTS_CURVE = list(range(0, 101, 10))
POINTS_EXTEND = list(range(0, 201, 10))
MSTYLE = [("auc", "AUC", "#1f77b4"), ("acc", "Accuracy", "#2ca02c"),
          ("bal", "Balanced acc", "#ff7f0e")]


# ------------------------------------------------------------------ loading
def load_base(conn, datasets):
    """Load L2-normed concatenated BASE features for the given datasets, aligned
    by sample id. Returns ids, X, y(raw), split, grp(patient), view, ds, patho."""
    ids0, mats = None, []
    for e in BASE:
        ids, v, _ = db.load_embeddings(conn, e, dataset=datasets)
        if ids0 is None:
            ids0 = ids
        else:
            assert np.array_equal(ids, ids0), f"embedding misalignment for {e}"
        mats.append(_l2(v))
    if ids0 is None or len(ids0) == 0:
        raise SystemExit(f"no embeddings found for {datasets}; run embed_all first")
    X = np.concatenate(mats, 1).astype(np.float32)
    meta = db.sample_meta(conn, ids0)
    ph = ",".join("?" * len(datasets))
    info = {r["id"]: r for r in conn.execute(
        f"SELECT id,split,pathology,abnormality FROM samples WHERE dataset IN ({ph})",
        datasets)}
    split = np.array([info[int(i)]["split"] for i in ids0])
    patho = np.array([str(info[int(i)]["pathology"]).upper() for i in ids0])
    abn = np.array([str(info[int(i)]["abnormality"]).lower() for i in ids0])
    y = np.array([meta[int(i)]["binary_label"] for i in ids0])
    view = np.array([meta[int(i)]["view"] for i in ids0])
    ds = np.array([meta[int(i)]["dataset"] for i in ids0])
    grp = np.array([meta[int(i)]["patient_id"] for i in ids0])
    keep = np.isin(view, ["CC", "MLO"])
    return (np.asarray(ids0)[keep], X[keep], y[keep], split[keep], grp[keep],
            view[keep], ds[keep], patho[keep], abn[keep])


def apply_task(task, ids, X, y, split, grp, view, ds, patho, abn, abn_filter=None):
    """Filter rows and (re)assign the binary label for the chosen task.
    abn_filter='mass' restricts to mass lesions (fair CBIS-mass comparison)."""
    # Mini-DDSM pathology = folder name upper-cased (BENIGN/CANCER/NORMAL); MIAS
    # (extend set) uses BENIGN/MALIGNANT. Treat CANCER and MALIGNANT both as positive.
    POS = ["CANCER", "MALIGNANT"]
    if task == "bm":
        m = np.isin(patho, ["BENIGN"] + POS)
    elif task == "nc":
        m = np.isin(patho, ["NORMAL"] + POS)
    else:
        raise ValueError(task)
    if abn_filter:
        m = m & (abn == abn_filter)               # keep only this lesion type
    ynew = np.isin(patho[m], POS).astype(int)     # positive = malignant/cancer
    return ids[m], X[m], ynew, split[m], grp[m], view[m], ds[m], patho[m], abn[m]


def load_ms(datasets):
    """Merge cached mammoscreen case scores (train+test npz) into ONE
    case_id -> score map, so runtime re-splitting is free."""
    full = {}
    for d in datasets:
        for sp in ("train", "test"):
            p = os.path.join(config.PROJECT_DIR, "cache", f"mammoscreen_{d}_{sp}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            full.update({str(c): float(s) for c, s in zip(z["cids"], z["s"])})
    return full


# ------------------------------------------------------------------ split helpers
def patient_test_mask(grp, y, test_frac, seed):
    uniq = np.unique(grp)
    plabel = {p: int(y[grp == p].max()) for p in uniq}
    rng = np.random.default_rng(seed)
    te = set()
    for cls in (0, 1):
        ps = [p for p in uniq if plabel[p] == cls]
        rng.shuffle(ps)
        te.update(ps[:int(round(test_frac * len(ps)))])
    return np.isin(grp, list(te))


def subsample_pool(grp, y, pool, frac, seed):
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


# ------------------------------------------------------------------ core eval
def fuse_full(ytr, Rtr, Mtr, yte, Rte, Mte, fix_w, thr_mode):
    """Percentile-fuse RAC + mammoscreen; return full metrics dict + test scores."""
    Rp, Mp = _pct(Rtr, Rtr), _pct(Mtr, Mtr)
    Rpe, Mpe = _pct(Rte, Rtr), _pct(Mte, Mtr)
    if fix_w is not None:
        w = float(fix_w)
    else:
        best = None
        for wv in np.linspace(0, 1, 21):
            a = roc_auc_score(ytr, wv * Rp + (1 - wv) * Mp)
            if best is None or a > best[0]:
                best = (a, wv)
        w = best[1]
    ftr, fte = w * Rp + (1 - w) * Mp, w * Rpe + (1 - w) * Mpe
    thr = _best_threshold(yte, fte) if thr_mode == "oracle" else _best_threshold(ytr, ftr)
    m = _metrics(yte, fte, thr)                       # acc,f1,sens,spec,tn..tp,bal_acc
    m["auc"] = roc_auc_score(yte, fte)
    m["w"], m["thr"] = float(w), float(thr)
    return m, fte, yte


def evaluate_once(conn, ids, X, y, grp, view, tr, te, ms_map, fix_w, thr_mode, seed):
    """One head-fit + fusion pass on the given train/test masks."""
    rcf_tr, rcf_te = head_case_scores(conn, ids, X, y, grp, view, tr, te, C_HEAD, seed)
    ytr, Rtr, Mtr, yte, Rte, Mte = align_B(rcf_tr, rcf_te, ms_map, ms_map)
    return fuse_full(ytr, Rtr, Mtr, yte, Rte, Mte, fix_w, thr_mode)


def ms_only(rcf_te, ms_map, thr_mode):
    """pt=0 baseline: B == mammoscreen score alone on the test cases."""
    cids = np.array([str(c) for c in rcf_te["cids"]])
    pidx = np.array(rcf_te["paired_idx"], dtype=int)
    keep = np.array([i for i in pidx if cids[i] in ms_map], dtype=int)
    yte = np.array(rcf_te["y_true"])[keep]
    Mte = np.array([ms_map[cids[i]] for i in keep])
    thr = _best_threshold(yte, Mte) if thr_mode == "oracle" else 0.5
    m = _metrics(yte, Mte, thr)
    m["auc"] = roc_auc_score(yte, Mte)
    m["w"], m["thr"] = 0.0, float(thr)
    return m


def agg_cm(mlist):
    """Sum confusion-matrix cells and average metrics over a list of metric dicts."""
    out = {k: float(np.mean([m[k] for m in mlist]))
           for k in ("auc", "accuracy", "bal_acc", "sensitivity", "specificity", "w")}
    out["std"] = {k: float(np.std([m[k] for m in mlist]))
                  for k in ("auc", "accuracy", "bal_acc")}
    for c in ("tn", "fp", "fn", "tp"):
        out[c] = int(np.sum([m[c] for m in mlist]))
    return out


def print_block(name, a, n_seeds):
    s = a["std"]
    print(f"\n=== {name} ===")
    print(f"  AUC        {a['auc']:.4f} ± {s['auc']:.4f}")
    print(f"  Accuracy   {a['accuracy']:.4f} ± {s['accuracy']:.4f}")
    print(f"  Bal. acc   {a['bal_acc']:.4f} ± {s['bal_acc']:.4f}")
    print(f"  Sens/Spec  {a['sensitivity']:.4f} / {a['specificity']:.4f}   (w={a['w']:.2f})")
    print(f"  Confusion matrix (summed over {n_seeds} runs):")
    print(f"                 pred 0   pred 1")
    print(f"     true 0  |   {a['tn']:6d}   {a['fp']:6d}")
    print(f"     true 1  |   {a['fn']:6d}   {a['tp']:6d}")


# ------------------------------------------------------------------ plotting
def combined_plot(out, pts, agg, title, xlabel, vline=None):
    plt.figure(figsize=(9, 5.5))
    for key, lab, col in MSTYLE:
        mean, std = agg[key]
        plt.plot(pts, mean, "-o", ms=4, color=col, label=lab)
        plt.fill_between(pts, mean - std, mean + std, color=col, alpha=0.13)
    if vline is not None:
        plt.axvline(vline, color="gray", ls="--", lw=1)
    plt.xlabel(xlabel)
    plt.ylabel("score")
    plt.title(title)
    plt.xticks(pts)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    print("saved", out)


# ------------------------------------------------------------------ drivers
def run_single(conn, data, ms_map, args):
    ids, X, y, _, grp, view, ds, patho, abn = data
    per = []
    for seed in SEEDS:
        te = patient_test_mask(grp, y, args.test_frac, seed)
        tr = ~te
        m, _, _ = evaluate_once(conn, ids, X, y, grp, view, tr, te, ms_map,
                                args.w, args.thr, seed)
        per.append(m)
        print(f"  seed={seed} tr={int(tr.sum())} te={int(te.sum())} "
              f"AUC={m['auc']:.4f} bal={m['bal_acc']:.4f} (w={m['w']:.2f})", flush=True)
    a = agg_cm(per)
    print_block(f"{args.task} single  test-frac={args.test_frac}  w={args.w}  thr={args.thr}",
                a, len(SEEDS))
    return a


def run_cv(conn, data, ms_map, args):
    ids, X, y, _, grp, view, ds, patho, abn = data
    sgkf = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=0)
    per = []
    for k, (trf, tef) in enumerate(sgkf.split(X, y, groups=grp)):
        te = np.zeros(len(y), bool); te[tef] = True
        tr = ~te
        m, _, _ = evaluate_once(conn, ids, X, y, grp, view, tr, te, ms_map,
                                args.w, args.thr, seed=0)
        per.append(m)
        print(f"  fold={k} te={int(te.sum())} AUC={m['auc']:.4f} bal={m['bal_acc']:.4f}",
              flush=True)
    a = agg_cm(per)
    print_block(f"{args.task} {args.kfold}-fold CV  w={args.w}  thr={args.thr}", a, args.kfold)
    return a


def run_curve_impl(conn, data, ms_map, args, points, extend):
    # curve/extend use a FIXED held-out test (seed 0) so the database grows under a
    # stable query set; DB subsampling varies with seed for the mean±std bands.
    ids, X, y, _, grp, view, ds, patho, abn = data
    extra = args.extra
    base_mask = ds != extra                      # Mini-DDSM (task rows); extra = MIAS
    # fixed held-out test on the base set (seed 0), stable across DB sizes
    te = patient_test_mask(grp, np.where(base_mask, y, 0), args.test_frac, 0) & base_mask
    pool_base = base_mask & ~te                  # Mini-DDSM train pool
    extra_pool = ds == extra                     # all MIAS -> head train only

    rows = []
    for pt in points:
        cf = min(pt, 100) / 100.0
        gf = max(pt - 100, 0) / 100.0
        for seed in SEEDS:
            if pt == 0:
                if seed != SEEDS[0]:
                    per0 = rows[-1]
                    rows.append(dict(per0)); rows[-1]["seed"] = seed
                    continue
                # mammoscreen-only baseline on the test cases
                tr_full = pool_base
                _, rcf_te = head_case_scores(conn, ids, X, y, grp, view, tr_full, te,
                                             C_HEAD, 0)
                m = ms_only(rcf_te, ms_map, args.thr)
                nimg = 0
            else:
                trm = subsample_pool(grp, y, pool_base, cf, seed)
                if extend and gf > 0:
                    trm = trm | subsample_pool(grp, y, extra_pool, gf, seed)
                m, _, _ = evaluate_once(conn, ids, X, y, grp, view, trm, te, ms_map,
                                        args.w, args.thr, seed)
                nimg = int(trm.sum())
            rows.append(dict(pt=pt, seed=seed, n_img=nimg, auc=m["auc"],
                             acc=m["accuracy"], bal=m["bal_acc"], w=m["w"],
                             sens=m["sensitivity"], spec=m["specificity"],
                             tn=m["tn"], fp=m["fp"], fn=m["fn"], tp=m["tp"]))
        last = [r for r in rows if r["pt"] == pt]
        print(f"pt={pt:3d} n_img~{int(np.mean([r['n_img'] for r in last])):4d} | "
              f"AUC={np.mean([r['auc'] for r in last]):.4f} "
              f"bal={np.mean([r['bal'] for r in last]):.4f}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.task}_{'extend' if extend else 'curve'}"
    csvp = os.path.join(args.outdir, f"miniddsm_B_{tag}.csv")
    with open(csvp, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print("saved", csvp)

    pts = np.array(points)
    agg = {}
    for key in ("auc", "acc", "bal"):
        mean = np.array([np.mean([r[key] for r in rows if r["pt"] == p]) for p in points])
        std = np.array([np.std([r[key] for r in rows if r["pt"] == p]) for p in points])
        agg[key] = (mean, std)
    if extend:
        title = (f"Ensembler B on Mini-DDSM ({args.task}) + {extra} extend "
                 f"(w={args.w}, {args.thr} thr)\nheld-out test, mean±std over {len(SEEDS)} seeds")
        xlabel = f"% of database   (0-100 = Mini-DDSM;  100-200 = full Mini-DDSM + % of {extra})"
        combined_plot(os.path.join(args.outdir, f"miniddsm_B_{tag}.png"), pts, agg,
                      title, xlabel, vline=100)
    else:
        title = (f"Ensembler B on Mini-DDSM ({args.task}) database scaling "
                 f"(w={args.w}, {args.thr} thr)\nheld-out test, mean±std over {len(SEEDS)} seeds")
        combined_plot(os.path.join(args.outdir, f"miniddsm_B_{tag}.png"), pts, agg,
                      title, "% of Mini-DDSM database used")
    # terminal-point confusion matrices
    for pt in ([100, 200] if extend else [100]):
        ml = [dict(accuracy=r["acc"], bal_acc=r["bal"], auc=r["auc"], w=r["w"],
                   sensitivity=r["sens"], specificity=r["spec"],
                   tn=r["tn"], fp=r["fp"], fn=r["fn"], tp=r["tp"])
              for r in rows if r["pt"] == pt]
        print_block(f"{args.task} {tag} @ {pt}%", agg_cm(ml), len(SEEDS))
    return rows


def run_cross(conn, args):
    """LEAKY exploratory: DB = Mini-DDSM (bm), test = CBIS paired test."""
    data = load_base(conn, ["miniddsm", "cbis_full"])
    ids, X, y, split, grp, view, ds, patho, abn = data
    # bm mass-only if requested; cbis_full is already mass-only benign/malignant
    m = ~((ds == "miniddsm") & (patho == "NORMAL"))
    if args.abn:
        m = m & ((ds == "cbis_full") | (abn == args.abn))
    ids, X, y, split, grp, view, ds, patho = (a[m] for a in
                                              (ids, X, y, split, grp, view, ds, patho))
    tr = ds == "miniddsm"                     # whole Mini-DDSM as gallery
    te = (ds == "cbis_full") & (split == "test")
    # mammoscreen map: Mini-DDSM (calibration) + CBIS (test scores) — BOTH needed
    cross_ms = load_ms(["miniddsm"])
    for sp in ("test", "train"):
        p = os.path.join(config.PROJECT_DIR, "cache", f"mammoscreen_{sp}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            cross_ms.update({str(c): float(s) for c, s in zip(z["cids"], z["s"])})
    m2, _, _ = evaluate_once(conn, ids, X, y, grp, view, tr, te, cross_ms,
                             args.w, args.thr, seed=0)
    a = agg_cm([m2])
    print_block("bm  CROSS: DB=Mini-DDSM  test=CBIS  [LEAKY]", a, 1)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["bm", "nc"], default="bm")
    ap.add_argument("--mode", choices=["single", "curve", "extend", "cv", "cross"],
                    default="single")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--w", type=float, default=0.6, help="fusion weight; omit-> pass -1 to sweep")
    ap.add_argument("--thr", choices=["train", "oracle"], default="train")
    ap.add_argument("--abn", choices=["mass", "calc"], default=None,
                    help="restrict to one lesion type (mass = fair CBIS-mass comparison)")
    ap.add_argument("--extra", default="mias_full", help="extend set (head train pool only)")
    ap.add_argument("--outdir", default=os.path.join(config.OUTPUT_DIR, "miniddsm"))
    args = ap.parse_args()
    if args.w is not None and args.w < 0:
        args.w = None                            # sweep w per config
    os.makedirs(args.outdir, exist_ok=True)

    conn = db.connect()
    if args.mode == "cross":
        run_cross(conn, args)
        return

    datasets = ["miniddsm"] if args.task == "bm" else ["miniddsm", "miniddsm_normal"]
    if args.mode == "extend":
        datasets = datasets + [args.extra]
    data = load_base(conn, datasets)
    data = apply_task(args.task, *data, abn_filter=args.abn)
    ms_map = load_ms(["miniddsm", "miniddsm_normal"])

    if args.mode == "single":
        run_single(conn, data, ms_map, args)
    elif args.mode == "cv":
        run_cv(conn, data, ms_map, args)
    elif args.mode == "curve":
        run_curve_impl(conn, data, ms_map, args, POINTS_CURVE, extend=False)
    elif args.mode == "extend":
        run_curve_impl(conn, data, ms_map, args, POINTS_EXTEND, extend=True)


if __name__ == "__main__":
    main()
