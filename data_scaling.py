"""Learning-curve / data-efficiency ablation for the two final ensemblers.

Question: as the *database* (our CBIS train pool) grows, which ensembler keeps
improving?  We subsample the train pool at 10/25/50/60/70/80/90/100 %, keep the
paired-151 test set FIXED, and re-run each ensembler fully-honestly at every
budget (heads re-fit with internal 5-fold CV, threshold picked out-of-fold,
fusion weights re-tuned on that budget's train only).  Multiple random seeds per
budget -> mean +- std error bands.

Two ensemblers (both fused in percentile space, weights tuned on train AUC):
  A (~0.9245): feat-head  +  RAC-head  +  mammoscreen-score      [3 branches]
  B (~0.894 ): RAC-head   +  mammoscreen-score                   [2 branches]

The mammoscreen SCORE branch is a frozen external model -> its per-case score is
invariant to how much of OUR data we use; only the trainable heads / weights /
threshold move.  So these curves isolate the data-sensitivity of the learnable
part of each ensembler.

Outputs: output/data_scaling_{auc,acc,balacc}.png, output/data_scaling.csv
Run:  python data_scaling.py
"""
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

FEAT = ["mammoscreen_feat"]
BASE = ["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"]
C_FEAT, C_BASE = 0.2, 0.5
FRACS = [0.10, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
SEEDS = [0, 1, 2, 3]
DATASET = "cbis_full"
# coarse honest weight grid for the 3-branch fusion (normalized on the fly)
WGRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]


# ---------------------------------------------------------------- data loading
def load_features(conn):
    """Load feat- and BASE-feature matrices ONCE (aligned by sample id)."""
    ids_f, Xf, y, split, pat, view = MV.load(conn, FEAT, DATASET)
    ids_b, Xb, y2, s2, p2, v2 = MV.load(conn, BASE, DATASET)
    assert np.array_equal(ids_f, ids_b), "feat/BASE sample ids misaligned"
    assert np.array_equal(y, y2)
    return ids_f, Xf, Xb, y, split, pat, view


def load_ms_maps():
    d_tr = np.load(os.path.join(config.PROJECT_DIR, "cache", "mammoscreen_train.npz"),
                   allow_pickle=True)
    d_te = np.load(os.path.join(config.PROJECT_DIR, "cache", "mammoscreen_test.npz"),
                   allow_pickle=True)
    ms_tr = {str(c): float(s) for c, s in zip(d_tr["cids"], d_tr["s"])}
    ms_te = {str(c): float(s) for c, s in zip(d_te["cids"], d_te["s"])}
    return ms_tr, ms_te


# ---------------------------------------------------------------- subsampling
def subsample_train(pat, y, split, frac, seed):
    """Patient-grouped, class-stratified subsample of the TRAIN pool.
    Drops whole patients; keeps the benign/malignant patient ratio; test stays
    untouched. Returns a boolean mask over all samples for the reduced train."""
    tr = split == "train"
    if frac >= 1.0:
        return tr.copy()
    train_pat = np.unique(pat[tr])
    plabel = {p: int(y[(pat == p) & tr].max()) for p in train_pat}  # any-malignant
    rng = np.random.default_rng(seed)
    keep = set()
    for cls in (0, 1):
        ps = [p for p in train_pat if plabel[p] == cls]
        rng.shuffle(ps)
        n = max(1, int(round(frac * len(ps))))
        keep.update(ps[:n])
    inkeep = np.array([p in keep for p in pat])
    return tr & inkeep


# ---------------------------------------------------------------- heads + fuse
def head_case_scores(conn, ids, X, y, pat, view, tr, te, C, seed):
    """Train the 3 multi-view heads (all/CC/MLO) on the `tr` mask, evaluate on
    `te`; return (train_oof_case_fuse, test_case_fuse) — the 4-way fusion."""
    cc, mlo = view == "CC", view == "MLO"
    oof_all, te_all = MV.head_cv(X[tr], y[tr], pat[tr], X[te], C, seed=seed)
    oof_cc, te_cc = MV.head_cv(X[tr & cc], y[tr & cc], pat[tr & cc], X[te & cc], C, seed=seed)
    oof_mlo, te_mlo = MV.head_cv(X[tr & mlo], y[tr & mlo], pat[tr & mlo], X[te & mlo], C, seed=seed)

    def branches(mask, a_scores, cc_scores, mlo_scores):
        ids_all = ids[mask]
        m_all = MV._map(ids_all, a_scores)
        cc_ids, mlo_ids = ids[mask & cc], ids[mask & mlo]
        return [
            dict(ids=cc_ids, score_pos=np.array([m_all[int(i)] for i in cc_ids])),
            dict(ids=mlo_ids, score_pos=np.array([m_all[int(i)] for i in mlo_ids])),
            dict(ids=cc_ids, score_pos=cc_scores),
            dict(ids=mlo_ids, score_pos=mlo_scores),
        ]

    tr_cf = evaluate.case_fuse(conn, branches(tr, oof_all, oof_cc, oof_mlo), method="mean")
    te_cf = evaluate.case_fuse(conn, branches(te, te_all, te_cc, te_mlo), method="mean")
    return tr_cf, te_cf


def _pct(x, ref):
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


def align(cf_f, cf_r, msmap):
    """Paired cases present in the mammoscreen map -> (y, F, R, M) arrays."""
    cids = np.array([str(c) for c in cf_f["cids"]])
    assert np.array_equal(cids, np.array([str(c) for c in cf_r["cids"]])), "cid misalign"
    y = np.array(cf_f["y_true"])
    pidx = np.array(cf_f["paired_idx"], dtype=int)
    keep = np.array([i for i in pidx if cids[i] in msmap], dtype=int)
    F = np.array(cf_f["y_score"])[keep]
    R = np.array(cf_r["y_score"])[keep]
    M = np.array([msmap[cids[i]] for i in keep])
    return y[keep], F, R, M


def _fuse_metrics(ytr, comps_tr, yte, comps_te, weights):
    """weights over percentile components -> AUC/ACC/balACC (honest threshold)."""
    ftr = sum(w * c for w, c in zip(weights, comps_tr))
    fte = sum(w * c for w, c in zip(weights, comps_te))
    thr = _best_threshold(ytr, ftr)
    m = _metrics(yte, fte, thr)
    return roc_auc_score(yte, fte), m["accuracy"], m["bal_acc"]


def ensembler_A(ytr, comps_tr, yte, comps_te):
    """3-branch: tune (wF,wR,wM) on train AUC, then eval on test."""
    best = None
    for wf in WGRID:
        for wr in WGRID:
            for wm in WGRID:
                if wf + wr + wm == 0:
                    continue
                wn = np.array([wf, wr, wm]) / (wf + wr + wm)
                ftr = wn[0] * comps_tr[0] + wn[1] * comps_tr[1] + wn[2] * comps_tr[2]
                a = roc_auc_score(ytr, ftr)
                if best is None or a > best[0]:
                    best = (a, wn)
    return _fuse_metrics(ytr, comps_tr, yte, comps_te, best[1])


def ensembler_B(ytr, comps_tr, yte, comps_te):
    """2-branch (RAC + mammoscreen-score): tune w on train AUC."""
    R_tr, M_tr = comps_tr[1], comps_tr[2]
    R_te, M_te = comps_te[1], comps_te[2]
    best = None
    for w in np.linspace(0, 1, 21):
        a = roc_auc_score(ytr, w * R_tr + (1 - w) * M_tr)
        if best is None or a > best[0]:
            best = (a, w)
    w = best[1]
    thr = _best_threshold(ytr, w * R_tr + (1 - w) * M_tr)
    fte = w * R_te + (1 - w) * M_te
    m = _metrics(yte, fte, thr)
    return roc_auc_score(yte, fte), m["accuracy"], m["bal_acc"]


# ---------------------------------------------------------------- one budget
def one_run(conn, data, ms_tr, ms_te, frac, seed):
    ids, Xf, Xb, y, split, pat, view = data
    trm = subsample_train(pat, y, split, frac, seed)
    te = split == "test"
    fcf_tr, fcf_te = head_case_scores(conn, ids, Xf, y, pat, view, trm, te, C_FEAT, seed)
    rcf_tr, rcf_te = head_case_scores(conn, ids, Xb, y, pat, view, trm, te, C_BASE, seed)

    ytr, Ftr, Rtr, Mtr = align(fcf_tr, rcf_tr, ms_tr)
    yte, Fte, Rte, Mte = align(fcf_te, rcf_te, ms_te)
    # percentile transform (reference = train components)
    ctr = [_pct(Ftr, Ftr), _pct(Rtr, Rtr), _pct(Mtr, Mtr)]
    cte = [_pct(Fte, Ftr), _pct(Rte, Rtr), _pct(Mte, Mtr)]

    A = ensembler_A(ytr, ctr, yte, cte)
    B = ensembler_B(ytr, ctr, yte, cte)
    n_train_cases = len(ytr)
    return A, B, n_train_cases


# ---------------------------------------------------------------- driver
def main():
    conn = db.connect()
    data = load_features(conn)
    ms_tr, ms_te = load_ms_maps()

    rows = []  # frac, seed, ntrain, A_auc,A_acc,A_bal, B_auc,B_acc,B_bal
    for frac in FRACS:
        for seed in SEEDS:
            if frac >= 1.0 and seed != SEEDS[0]:
                # full data is identical across subsample seeds; still vary the
                # CV fold seed to get honest variance, so keep running.
                pass
            A, B, ntr = one_run(conn, data, ms_tr, ms_te, frac, seed)
            rows.append(dict(frac=frac, seed=seed, ntrain=ntr,
                             A_auc=A[0], A_acc=A[1], A_bal=A[2],
                             B_auc=B[0], B_acc=B[1], B_bal=B[2]))
            print(f"frac={frac:.2f} seed={seed} ntrain={ntr:4d} | "
                  f"A auc={A[0]:.4f} acc={A[1]:.4f} bal={A[2]:.4f} | "
                  f"B auc={B[0]:.4f} acc={B[1]:.4f} bal={B[2]:.4f}", flush=True)

    # persist raw
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(config.OUTPUT_DIR, "data_scaling.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # aggregate mean/std per frac
    fr = np.array(FRACS)
    agg = {}
    for metric in ["auc", "acc", "bal"]:
        for ens in ["A", "B"]:
            key = f"{ens}_{metric}"
            mean = np.array([np.mean([r[key] for r in rows if r["frac"] == fc]) for fc in FRACS])
            std = np.array([np.std([r[key] for r in rows if r["frac"] == fc]) for fc in FRACS])
            agg[key] = (mean, std)

    titles = {"auc": "Case-level AUC", "acc": "Accuracy", "bal": "Balanced accuracy"}
    for metric in ["auc", "acc", "bal"]:
        plt.figure(figsize=(7, 5))
        for ens, color, label in [("A", "#1f77b4", "Ensembler A (feat+RAC+score, ~0.92)"),
                                   ("B", "#d62728", "Ensembler B (RAC+score, ~0.89)")]:
            mean, std = agg[f"{ens}_{metric}"]
            plt.plot(fr * 100, mean, "-o", color=color, label=label)
            plt.fill_between(fr * 100, mean - std, mean + std, color=color, alpha=0.15)
        plt.xlabel("% of training database used")
        plt.ylabel(titles[metric])
        plt.title(f"{titles[metric]} vs. database size (paired-151 test, mean±std over {len(SEEDS)} seeds)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = os.path.join(config.OUTPUT_DIR, f"data_scaling_{metric}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=140)
        plt.close()
        print("saved", out)

    # verdict: slope + delta(100-50) on AUC
    print("\n=== data-sensitivity (AUC) ===")
    for ens in ["A", "B"]:
        mean, _ = agg[f"{ens}_auc"]
        i50 = FRACS.index(0.50)
        slope = np.polyfit(fr, mean, 1)[0]
        print(f"  Ensembler {ens}: AUC 50%={mean[i50]:.4f} -> 100%={mean[-1]:.4f} "
              f"(delta={mean[-1]-mean[i50]:+.4f}), linear slope/100%data={slope:.4f}")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
