"""Ensembler B — ORDER / PATH-DEPENDENCE ablation of the database learning curve.

data_scaling_extend.py grows the database in one fixed order (all CBIS first, then
the extra set). Any gain in the 100->200 half is therefore confounded with POSITION
on the curve ("adding N images to an already-saturated model") rather than with the
identity of the data. This script re-runs the same curve under different growth
orders so composition can be separated from ordering:

  --order cbis_first   0..100 = % of CBIS;  100..200 = full CBIS + % of extra   [EXP0]
  --order extra_first  0..100 = % of extra; 100..200 = full extra + % of CBIS   [EXP1]
  --order mixed        pt = pt/2 % of the POOLED CBIS+extra database, drawn at
                       random from both (class-stratified, patient/case-grouped)  [EXP2]

All three orders share the same two endpoints (pt=0 -> no database; pt=200 -> the
full CBIS+extra pool), so the curves are directly comparable, and n_img in the CSV
lets them be read at MATCHED image counts.

Two protocol changes vs data_scaling_extend.py, both forced by EXP1:

  * CALIBRATION SET. The percentile reference / fusion weight / threshold must live
    on CBIS-train (the test domain), but in EXP1's first half no CBIS image is in the
    head's fit pool. So calibration always uses ALL CBIS-train paired cases, scored
    out-of-fold when the image is in the fit pool and by the fold-ensemble when it is
    not (i.e. genuinely held out). From pt=100 onward under cbis_first this is
    identical to the old behaviour.
  * VIEW-SPECIALIST FALLBACK. MIAS is MLO-only, so a CC-specialist cannot be fit from
    a MIAS-only pool. When a view-specialist has no usable training data its branch
    falls back to the ALL head (the 4-way fusion degenerates gracefully to 2 unique
    branches). Reported per point as `fb` in the log.

Test set is frozen at paired-151 CBIS. The extra set only ever enters the RAC head's
training pool; the frozen mammoscreen SCORE branch is CBIS-only throughout.

Run (paper config = w 0.6, C 0.2, oracle threshold):
  python data_scaling_order.py --order extra_first --extra mias_full --tag mias \
      --label "MIAS (film)" --w 0.6 --c 0.2 --thr oracle --outdir output/exp1_mias_first
  python data_scaling_order.py --order mixed --extra mias_full --tag mias \
      --label "MIAS (film)" --w 0.6 --c 0.2 --thr oracle --outdir output/exp2_mixed
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
import evaluate
import multiview_probe as MV
from mil import _best_threshold, _metrics
from data_scaling import load_ms_maps, _pct

BASE = ["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"]
C_BASE = 0.5
CBIS_DS = "cbis_full"
EX_DS = "inbreast_full"          # overwritten by --extra at runtime
POINTS = list(range(0, 201, 10))
SEEDS = [0, 1, 2, 3]
N_SPLITS = 5


# ---------------------------------------------------------------- data loading
def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def load_base_combined(conn):
    """BASE features for CBIS + the extra set, aligned by sample id. Groups are
    prefixed with the dataset so CBIS patient_ids can never collide with extra
    case_ids once the two pools are shuffled together (EXP2)."""
    ids0, mats = None, []
    for e in BASE:
        ids, v, _ = db.load_embeddings(conn, e, dataset=[CBIS_DS, EX_DS])
        if ids0 is None:
            ids0 = ids
        else:
            assert np.array_equal(ids, ids0), f"misalign {e}"
        mats.append(_l2(v))
    X = np.concatenate(mats, 1).astype(np.float32)
    meta = db.sample_meta(conn, ids0)
    info = {r["id"]: r for r in conn.execute(
        "SELECT id,split FROM samples WHERE dataset IN (?,?)", (CBIS_DS, EX_DS))}
    split = np.array([info[int(i)]["split"] for i in ids0])
    y = np.array([meta[int(i)]["binary_label"] for i in ids0])
    view = np.array([meta[int(i)]["view"] for i in ids0])
    ds = np.array([meta[int(i)]["dataset"] for i in ids0])
    grp = np.array([f"{meta[int(i)]['dataset']}:"
                    f"{meta[int(i)]['case_id'] if meta[int(i)]['dataset'] == EX_DS else meta[int(i)]['patient_id']}"
                    for i in ids0])
    keep = np.isin(view, ["CC", "MLO"])                    # drop the lone FB image
    return (np.asarray(ids0)[keep], X[keep], y[keep], split[keep],
            grp[keep], view[keep], ds[keep])


# ---------------------------------------------------------------- subsampling
def _pick(y, grp, mask, frac, rng):
    """Class-stratified, group-respecting subsample of `mask` down to `frac`."""
    if frac >= 1.0:
        return mask.copy()
    if frac <= 0.0:
        return np.zeros_like(mask)
    uniq = np.unique(grp[mask])
    glabel = {g: int(y[mask & (grp == g)].max()) for g in uniq}
    keep = set()
    for cls in (0, 1):
        gs = [g for g in uniq if glabel[g] == cls]
        rng.shuffle(gs)
        n = max(1, int(round(frac * len(gs))))
        keep.update(gs[:n])
    return mask & np.isin(grp, list(keep))


def _pick_pooled(y, grp, mask, frac, rng):
    """EXP2: draw groups at RANDOM from the pooled (CBIS+extra) train set until the
    image budget is met. Class-stratified; dataset membership is left to chance, so
    the mix fluctuates around the natural CBIS:extra ratio."""
    if frac >= 1.0:
        return mask.copy()
    if frac <= 0.0:
        return np.zeros_like(mask)
    uniq = np.unique(grp[mask])
    glabel, gsize = {}, {}
    for g in uniq:
        m = mask & (grp == g)
        glabel[g], gsize[g] = int(y[m].max()), int(m.sum())
    keep = []
    for cls in (0, 1):
        gs = [g for g in uniq if glabel[g] == cls]
        rng.shuffle(gs)
        target, acc = frac * sum(gsize[g] for g in gs), 0
        for g in gs:
            if acc >= target:
                break
            keep.append(g)
            acc += gsize[g]
    return mask & np.isin(grp, keep)


def build_train_mask(y, split, ds, grp, pt, order, seed):
    """Fit-pool mask for a given curve point under a given growth order."""
    tr = split == "train"
    cbis_tr, ex_tr = tr & (ds == CBIS_DS), tr & (ds == EX_DS)
    rng = np.random.default_rng(1000 * (0 if order == "cbis_first" else
                                        1 if order == "extra_first" else 2) + seed)
    if order == "mixed":
        return _pick_pooled(y, grp, tr, pt / 200.0, rng)
    if order == "cbis_first":
        a, b = (pt / 100.0, 0.0) if pt <= 100 else (1.0, (pt - 100) / 100.0)
    else:                                     # extra_first
        b, a = (pt / 100.0, 0.0) if pt <= 100 else (1.0, (pt - 100) / 100.0)
    return _pick(y, grp, cbis_tr, a, rng) | _pick(y, grp, ex_tr, b, rng)


# ---------------------------------------------------------------- heads + fuse
def _head(Xf, yf, gf, Xe, C, seed):
    """head_cv guarded against degenerate fit pools -> (oof, eval_pred) or (None, None)."""
    if len(yf) < 2 * N_SPLITS or len(np.unique(yf)) < 2 or len(np.unique(gf)) < N_SPLITS:
        return None, None
    return MV.head_cv(Xf, yf, gf, Xe, C, seed=seed, n_splits=N_SPLITS)


def head_case_scores(conn, ids, X, y, grp, view, fit, calib, te, C, seed):
    """Score every calibration + test image. Images inside the fit pool get their
    out-of-fold score; images outside it get the fold-ensemble prediction."""
    cc, mlo = view == "CC", view == "MLO"
    ev = (calib & ~fit) | te                       # need out-of-sample predictions
    n = len(ids)

    S_all = np.full(n, np.nan)
    oof, pe = _head(X[fit], y[fit], grp[fit], X[ev], C, seed)
    if oof is None:
        raise RuntimeError("ALL head has no usable training data at this point")
    S_all[fit], S_all[ev] = oof, pe

    fb = []
    def spec(m, name):
        f, e = fit & m, ev & m
        o, p = _head(X[f], y[f], grp[f], X[e], C, seed)
        if o is None:                              # e.g. MIAS-only pool has no CC
            fb.append(name)
            return S_all
        S = np.full(n, np.nan)
        S[f], S[e] = o, p
        return S

    S_cc, S_mlo = spec(cc, "CC"), spec(mlo, "MLO")

    def branches(mask):
        cid, mid = ids[mask & cc], ids[mask & mlo]
        return [dict(ids=cid, score_pos=S_all[mask & cc]),
                dict(ids=mid, score_pos=S_all[mask & mlo]),
                dict(ids=cid, score_pos=S_cc[mask & cc]),
                dict(ids=mid, score_pos=S_mlo[mask & mlo])]

    tr_cf = evaluate.case_fuse(conn, branches(calib), method="mean")
    te_cf = evaluate.case_fuse(conn, branches(te), method="mean")
    return tr_cf, te_cf, fb


def align_B(rcf_tr, rcf_te, ms_tr, ms_te):
    def one(cf, msmap):
        cids = np.array([str(c) for c in cf["cids"]])
        pidx = np.array(cf["paired_idx"], dtype=int)
        keep = np.array([i for i in pidx if cids[i] in msmap], dtype=int)
        return (np.array(cf["y_true"])[keep], np.array(cf["y_score"])[keep],
                np.array([msmap[cids[i]] for i in keep]))
    ytr, Rtr, Mtr = one(rcf_tr, ms_tr)
    yte, Rte, Mte = one(rcf_te, ms_te)
    return ytr, Rtr, Mtr, yte, Rte, Mte


def ensembler_B(ytr, Rtr, Mtr, yte, Rte, Mte, fix_w=None, thr_mode="train"):
    """Percentile-fuse RAC head + mammoscreen score (identical to data_scaling_extend)."""
    Rp, Mp = _pct(Rtr, Rtr), _pct(Mtr, Mtr)
    Rpe, Mpe = _pct(Rte, Rtr), _pct(Mte, Mtr)
    if fix_w is not None:
        w = float(fix_w)
    else:
        best = None
        for w in np.linspace(0, 1, 21):
            a = roc_auc_score(ytr, w * Rp + (1 - w) * Mp)
            if best is None or a > best[0]:
                best = (a, w)
        w = best[1]
    fte = w * Rpe + (1 - w) * Mpe
    thr = _best_threshold(yte, fte) if thr_mode == "oracle" \
        else _best_threshold(ytr, w * Rp + (1 - w) * Mp)
    m = _metrics(yte, fte, thr)
    return roc_auc_score(yte, fte), m["accuracy"], m["bal_acc"], float(w)


# ---------------------------------------------------------------- driver
ORDER_TITLE = {
    "cbis_first":  "EXP0: CBIS first, then {label}",
    "extra_first": "EXP1: {label} first, then CBIS",
    "mixed":       "EXP2: CBIS + {label} drawn at random",
}


def axis_label(order, label):
    if order == "cbis_first":
        return f"% of database used   (0-100 = CBIS;  100-200 = full CBIS + % of {label})"
    if order == "extra_first":
        return f"% of database used   (0-100 = {label};  100-200 = all {label} + % of CBIS)"
    return f"% of database used   (x/2 = % of the pooled CBIS + {label} set, drawn at random)"


MSTYLE = [("auc", "AUC", "#1f77b4"), ("acc", "Accuracy", "#2ca02c"),
          ("bal", "Balanced acc", "#ff7f0e")]


def combined_plot(out, pts, agg, order, label, fix_w, thr_mode):
    """All three metrics on ONE axes — the same view as output/combined/B_*.png."""
    plt.figure(figsize=(9, 5.5))
    for m, mlabel, color in MSTYLE:
        mean, std = agg[m]
        plt.plot(pts, mean, "-o", ms=4, color=color, label=mlabel)
        plt.fill_between(pts, mean - std, mean + std, color=color, alpha=0.13)
    ymin = min(agg[m][0].min() for m, _, _ in MSTYLE)
    if order == "cbis_first":
        plt.axvline(100, color="gray", ls="--", lw=1)
        plt.text(50, ymin, "CBIS (film)", color="gray", fontsize=9, ha="center", va="bottom")
        plt.text(101, ymin, f"  + {label} -->", color="gray", fontsize=9, va="bottom")
    elif order == "extra_first":
        plt.axvline(100, color="gray", ls="--", lw=1)
        plt.text(50, ymin, label, color="gray", fontsize=9, ha="center", va="bottom")
        plt.text(101, ymin, "  + CBIS (film) -->", color="gray", fontsize=9, va="bottom")
    else:
        plt.text(100, ymin, f"random mix of CBIS + {label}", color="gray",
                 fontsize=9, ha="center", va="bottom")
    plt.xlabel(axis_label(order, label))
    plt.ylabel("score")
    plt.title(f"Ensembler B (RAC + mammoscreen, w={fix_w}, {thr_mode}-threshold)  —  "
              f"{ORDER_TITLE[order].format(label=label)}\n"
              f"(paired-151 test, mean±std over {len(SEEDS)} seeds)")
    plt.xticks(pts)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def main():
    global EX_DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=["cbis_first", "extra_first", "mixed"],
                    default="extra_first")
    ap.add_argument("--extra", default="mias_full")
    ap.add_argument("--tag", default="mias")
    ap.add_argument("--label", default="MIAS (film)")
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--c", type=float, default=C_BASE)
    ap.add_argument("--thr", choices=["train", "oracle"], default="train")
    ap.add_argument("--outdir", default=config.OUTPUT_DIR)
    args = ap.parse_args()
    EX_DS = args.extra
    order, tag, label = args.order, args.tag, args.label
    fix_w, thr_mode, C_head, outdir = args.w, args.thr, args.c, args.outdir

    conn = db.connect()
    ids, X, y, split, grp, view, ds = load_base_combined(conn)
    ms_tr, ms_te = load_ms_maps()
    te = split == "test"
    calib = (split == "train") & (ds == CBIS_DS)      # always the full CBIS train pool
    print(f"order={order} extra={EX_DS} | CBIS-train {int(calib.sum())} imgs, "
          f"extra-train {int(((split=='train')&(ds==EX_DS)).sum())} imgs, test {int(te.sum())} imgs")

    rows = []
    for pt in POINTS:
        for seed in SEEDS:
            if pt == 0:
                if seed != SEEDS[0]:
                    continue
                fit0 = build_train_mask(y, split, ds, grp, 200, "cbis_first", 0)
                _, rcf_te, _ = head_case_scores(conn, ids, X, y, grp, view,
                                                fit0, calib, te, C_head, 0)
                cids = np.array([str(c) for c in rcf_te["cids"]])
                pidx = np.array(rcf_te["paired_idx"], dtype=int)
                keep = np.array([i for i in pidx if cids[i] in ms_te], dtype=int)
                yte_p = np.array(rcf_te["y_true"])[keep]
                Mte_p = np.array([ms_te[cids[i]] for i in keep])
                auc = roc_auc_score(yte_p, Mte_p)
                thr0 = _best_threshold(yte_p, Mte_p) if thr_mode == "oracle" else 0.5
                mm = _metrics(yte_p, Mte_p, thr0)
                for s in SEEDS:
                    rows.append(dict(pt=pt, seed=s, n_cbis=0, n_ex=0, n_img=0,
                                     w=0.0, auc=auc, acc=mm["accuracy"], bal=mm["bal_acc"]))
                print(f"pt={pt:3d} seed=all  DB=0             | B(mammoscreen only) "
                      f"auc={auc:.4f} acc={mm['accuracy']:.4f} bal={mm['bal_acc']:.4f}", flush=True)
                continue

            fit = build_train_mask(y, split, ds, grp, pt, order, seed)
            n_cbis = int((fit & (ds == CBIS_DS)).sum())
            n_ex = int((fit & (ds == EX_DS)).sum())
            rcf_tr, rcf_te, fb = head_case_scores(conn, ids, X, y, grp, view,
                                                  fit, calib, te, C_head, seed)
            ytr, Rtr, Mtr, yte, Rte, Mte = align_B(rcf_tr, rcf_te, ms_tr, ms_te)
            auc, acc, bal, w = ensembler_B(ytr, Rtr, Mtr, yte, Rte, Mte,
                                           fix_w=fix_w, thr_mode=thr_mode)
            rows.append(dict(pt=pt, seed=seed, n_cbis=n_cbis, n_ex=n_ex,
                             n_img=n_cbis + n_ex, w=w, auc=auc, acc=acc, bal=bal))
            print(f"pt={pt:3d} seed={seed} DB={n_cbis:4d}cbis+{n_ex:4d}ex | "
                  f"B auc={auc:.4f} acc={acc:.4f} bal={bal:.4f}"
                  f"{'  fallback:' + '/'.join(fb) if fb else ''}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"data_scaling_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    pts = np.array(POINTS)
    agg = {m: (np.array([np.mean([r[m] for r in rows if r["pt"] == p]) for p in POINTS]),
               np.array([np.std([r[m] for r in rows if r["pt"] == p]) for p in POINTS]))
           for m in ["auc", "acc", "bal"]}
    nimg = np.array([np.mean([r["n_img"] for r in rows if r["pt"] == p]) for p in POINTS])

    titles = {"auc": "Case-level AUC", "acc": "Accuracy", "bal": "Balanced accuracy"}
    for metric in ["auc", "acc", "bal"]:
        mean, std = agg[metric]
        plt.figure(figsize=(9, 5))
        plt.plot(pts, mean, "-o", color="#d62728", label="Ensembler B (RAC + mammoscreen-score)")
        plt.fill_between(pts, mean - std, mean + std, color="#d62728", alpha=0.15)
        if order != "mixed":
            plt.axvline(100, color="gray", ls="--", lw=1)
        plt.xlabel(axis_label(order, label))
        plt.ylabel(titles[metric])
        plt.title(f"{titles[metric]} vs. database size — order={order}, extra={label}\n"
                  f"(paired-151 test, mean±std over {len(SEEDS)} seeds)")
        plt.xticks(pts)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        out = os.path.join(outdir, f"data_scaling_{tag}_{metric}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=140)
        plt.close()
        print("saved", out)

    # combined single-axes image (same view as output/combined) + aggregated csv
    comb = os.path.join(outdir, f"B_{tag}_{order}.png")
    combined_plot(comb, pts, agg, order, label, fix_w, thr_mode)

    agg_csv = os.path.join(outdir, f"B_{tag}_{order}.csv")
    with open(agg_csv, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["pt", "n_img", "auc_mean", "auc_std", "acc_mean", "acc_std",
                      "bal_mean", "bal_std"])
        for i, p in enumerate(POINTS):
            wtr.writerow([p, f"{nimg[i]:.1f}"] +
                         [f"{agg[m][j][i]:.6f}" for m in ["auc", "acc", "bal"] for j in (0, 1)])
    print("saved", comb, "+", agg_csv)

    ma, mb = agg["auc"][0], agg["bal"][0]
    i100 = POINTS.index(100)
    print(f"\n=== Ensembler B  order={order}  extra={label} ===")
    print(f"  0%    AUC={ma[0]:.4f}  bal={mb[0]:.4f}   [db 0 imgs]")
    print(f"  100%  AUC={ma[i100]:.4f}  bal={mb[i100]:.4f}   [db~{int(nimg[i100])} imgs]")
    print(f"  200%  AUC={ma[-1]:.4f}  bal={mb[-1]:.4f}   [db~{int(nimg[-1])} imgs]")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
