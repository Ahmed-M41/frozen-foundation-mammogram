"""Ensembler B only: database learning curve extended past 100% with InBreast.

x-axis (user-requested): 0,10,...,100,110,...,200.
  0..100   : fraction of the CBIS train pool used (film).      InBreast NOT used.
  100..200 : full CBIS train + fraction of InBreast added.     g=(x-100)/100.
             x=200 => full CBIS + ALL InBreast.

NOTE: InBreast (~409 CC/MLO imgs) is only ~33% of CBIS-train in image count, so
the second half is a CONCEPTUAL axis ("% of InBreast added"), not a true doubling.
Actual database image counts are annotated on the plot and saved in the CSV.

InBreast (digital) augments the RAC head's TRAINING pool only; it has no CC/MLO
pairing and no patient grouping, so each image gets a unique CV group. All
calibration (threshold + fusion weight) stays on CBIS (film) — the test domain.
Test set is frozen at paired-151 CBIS. Frozen mammoscreen SCORE branch is applied
only to CBIS cases (it never sees InBreast).

Ensembler B (percentile fusion, w tuned on CBIS-train AUC):  RAC-head + mammoscreen-score
0% point: no database -> B falls back to the frozen mammoscreen score alone.

Outputs: output/data_scaling_ib_{auc,acc,bal}.png, output/data_scaling_ib.csv
Run:  python data_scaling_inbreast.py
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
from data_scaling import load_ms_maps, _pct

BASE = ["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"]
C_BASE = 0.5
CBIS_DS = "cbis_full"
IB_DS = "inbreast_full"
POINTS = list(range(0, 201, 10))          # 0,10,...,200
SEEDS = [0, 1, 2, 3]


# ---------------------------------------------------------------- data loading
def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def load_base_combined(conn):
    """Load BASE features for CBIS + InBreast together (aligned by sample id).
    Returns ids, X, y, split, grp, view, ds. grp = patient_id (CBIS) or unique
    case_id (InBreast). Non-CC/MLO views dropped."""
    ids0, mats = None, []
    for e in BASE:
        ids, v, _ = db.load_embeddings(conn, e, dataset=[CBIS_DS, IB_DS])
        if ids0 is None:
            ids0 = ids
        else:
            assert np.array_equal(ids, ids0), f"misalign {e}"
        mats.append(_l2(v))
    X = np.concatenate(mats, 1).astype(np.float32)
    meta = db.sample_meta(conn, ids0)
    info = {r["id"]: r for r in conn.execute(
        "SELECT id,split FROM samples WHERE dataset IN (?,?)", (CBIS_DS, IB_DS))}
    split = np.array([info[int(i)]["split"] for i in ids0])
    y = np.array([meta[int(i)]["binary_label"] for i in ids0])
    view = np.array([meta[int(i)]["view"] for i in ids0])
    ds = np.array([meta[int(i)]["dataset"] for i in ids0])
    grp = np.array([meta[int(i)]["case_id"] if meta[int(i)]["dataset"] == IB_DS
                    else meta[int(i)]["patient_id"] for i in ids0])
    keep = np.isin(view, ["CC", "MLO"])                    # drop the lone FB image
    return (np.asarray(ids0)[keep], X[keep], y[keep], split[keep],
            grp[keep], view[keep], ds[keep])


# ---------------------------------------------------------------- subsampling
def build_train_mask(y, split, ds, grp, cbis_frac, ib_frac, seed):
    """Patient-stratified CBIS subsample (film) OR whole-image InBreast subsample
    (digital), unioned into one training mask. Test stays untouched."""
    tr = split == "train"
    cbis_tr = tr & (ds == CBIS_DS)
    ib_tr = tr & (ds == IB_DS)
    rng = np.random.default_rng(seed)

    def pick(mask, frac, group_by_patient):
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

    return pick(cbis_tr, cbis_frac, True) | pick(ib_tr, ib_frac, False)


# ---------------------------------------------------------------- heads + fuse
def head_case_scores(conn, ids, X, y, grp, view, tr, te, C, seed):
    cc, mlo = view == "CC", view == "MLO"
    oof_all, te_all = MV.head_cv(X[tr], y[tr], grp[tr], X[te], C, seed=seed)
    oof_cc, te_cc = MV.head_cv(X[tr & cc], y[tr & cc], grp[tr & cc], X[te & cc], C, seed=seed)
    oof_mlo, te_mlo = MV.head_cv(X[tr & mlo], y[tr & mlo], grp[tr & mlo], X[te & mlo], C, seed=seed)

    def branches(mask, a_scores, cc_scores, mlo_scores):
        m_all = MV._map(ids[mask], a_scores)
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


def align_B(rcf_tr, rcf_te, ms_tr, ms_te):
    """Keep paired CBIS cases present in the mammoscreen map (drops InBreast)."""
    def one(cf, msmap):
        cids = np.array([str(c) for c in cf["cids"]])
        pidx = np.array(cf["paired_idx"], dtype=int)
        keep = np.array([i for i in pidx if cids[i] in msmap], dtype=int)
        return (np.array(cf["y_true"])[keep], np.array(cf["y_score"])[keep],
                np.array([msmap[cids[i]] for i in keep]))
    ytr, Rtr, Mtr = one(rcf_tr, ms_tr)
    yte, Rte, Mte = one(rcf_te, ms_te)
    return ytr, Rtr, Mtr, yte, Rte, Mte


def ensembler_B(ytr, Rtr, Mtr, yte, Rte, Mte):
    """Percentile-fuse RAC head + mammoscreen score; tune w on CBIS-train AUC."""
    Rp, Mp = _pct(Rtr, Rtr), _pct(Mtr, Mtr)
    Rpe, Mpe = _pct(Rte, Rtr), _pct(Mte, Mtr)
    best = None
    for w in np.linspace(0, 1, 21):
        a = roc_auc_score(ytr, w * Rp + (1 - w) * Mp)
        if best is None or a > best[0]:
            best = (a, w)
    w = best[1]
    thr = _best_threshold(ytr, w * Rp + (1 - w) * Mp)
    fte = w * Rpe + (1 - w) * Mpe
    m = _metrics(yte, fte, thr)
    return roc_auc_score(yte, fte), m["accuracy"], m["bal_acc"], float(w)


# ---------------------------------------------------------------- driver
def point_to_fracs(pt):
    if pt <= 100:
        return pt / 100.0, 0.0
    return 1.0, (pt - 100) / 100.0


def main():
    conn = db.connect()
    ids, X, y, split, grp, view, ds = load_base_combined(conn)
    ms_tr, ms_te = load_ms_maps()
    te = split == "test"

    rows = []
    for pt in POINTS:
        cf, gf = point_to_fracs(pt)
        for seed in SEEDS:
            if pt == 0:
                # no head: B = pure mammoscreen score (deterministic; 1 seed only)
                if seed != SEEDS[0]:
                    continue
                tr_full = build_train_mask(y, split, ds, grp, 1.0, 0.0, 0)
                _, rcf_te = head_case_scores(conn, ids, X, y, grp, view, tr_full, te, C_BASE, 0)
                # reuse alignment just to get yte + Mte on paired-151
                cids = np.array([str(c) for c in rcf_te["cids"]])
                pidx = np.array(rcf_te["paired_idx"], dtype=int)
                keep = np.array([i for i in pidx if cids[i] in ms_te], dtype=int)
                yte_p = np.array(rcf_te["y_true"])[keep]
                Mte_p = np.array([ms_te[cids[i]] for i in keep])
                auc = roc_auc_score(yte_p, Mte_p)
                pred = (Mte_p >= 0.5).astype(int)
                mm = _metrics(yte_p, Mte_p, 0.5)
                nimg = 0
                for s in SEEDS:  # replicate flat 0% across seeds for clean bands
                    rows.append(dict(pt=pt, seed=s, n_cbis=0, n_ib=0, n_img=0,
                                     w=0.0, auc=auc, acc=mm["accuracy"], bal=mm["bal_acc"]))
                print(f"pt={pt:3d} seed=all  DB=0            | B(mammoscreen only) "
                      f"auc={auc:.4f} acc={mm['accuracy']:.4f} bal={mm['bal_acc']:.4f}", flush=True)
                continue

            trm = build_train_mask(y, split, ds, grp, cf, gf, seed)
            n_cbis = int(((trm) & (ds == CBIS_DS)).sum())
            n_ib = int(((trm) & (ds == IB_DS)).sum())
            rcf_tr, rcf_te = head_case_scores(conn, ids, X, y, grp, view, trm, te, C_BASE, seed)
            ytr, Rtr, Mtr, yte, Rte, Mte = align_B(rcf_tr, rcf_te, ms_tr, ms_te)
            auc, acc, bal, w = ensembler_B(ytr, Rtr, Mtr, yte, Rte, Mte)
            rows.append(dict(pt=pt, seed=seed, n_cbis=n_cbis, n_ib=n_ib,
                             n_img=n_cbis + n_ib, w=w, auc=auc, acc=acc, bal=bal))
            print(f"pt={pt:3d} seed={seed} DB={n_cbis:4d}cbis+{n_ib:3d}ib | "
                  f"B auc={auc:.4f} acc={acc:.4f} bal={bal:.4f} (w={w:.2f})", flush=True)

    # persist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(config.OUTPUT_DIR, "data_scaling_ib.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    # aggregate
    pts = np.array(POINTS)
    agg = {}
    for metric in ["auc", "acc", "bal"]:
        mean = np.array([np.mean([r[metric] for r in rows if r["pt"] == p]) for p in POINTS])
        std = np.array([np.std([r[metric] for r in rows if r["pt"] == p]) for p in POINTS])
        agg[metric] = (mean, std)
    # mean image count per point (for annotation)
    nimg = np.array([np.mean([r["n_img"] for r in rows if r["pt"] == p]) for p in POINTS])

    titles = {"auc": "Case-level AUC", "acc": "Accuracy", "bal": "Balanced accuracy"}
    for metric in ["auc", "acc", "bal"]:
        mean, std = agg[metric]
        plt.figure(figsize=(9, 5))
        plt.plot(pts, mean, "-o", color="#d62728", label="Ensembler B (RAC + mammoscreen-score)")
        plt.fill_between(pts, mean - std, mean + std, color="#d62728", alpha=0.15)
        plt.axvline(100, color="gray", ls="--", lw=1)
        ymid = mean.min() + 0.5 * (mean.max() - mean.min())
        plt.text(101, mean.min(), "  + InBreast (digital) added -->", color="gray", fontsize=9, va="bottom")
        plt.text(50, mean.min(), "CBIS (film)", color="gray", fontsize=9, ha="center", va="bottom")
        plt.xlabel("% of database used   (0-100 = CBIS-train;  100-200 = full CBIS + % of InBreast)")
        plt.ylabel(titles[metric])
        plt.title(f"{titles[metric]} vs. database size, extended with InBreast "
                  f"(paired-151 test, mean±std over {len(SEEDS)} seeds)")
        plt.xticks(pts, rotation=0)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        out = os.path.join(config.OUTPUT_DIR, f"data_scaling_ib_{metric}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=140)
        plt.close()
        print("saved", out)

    # verdict
    m, _ = agg["auc"]
    i100 = POINTS.index(100)
    print("\n=== Ensembler B: does adding InBreast beyond 100% help? (AUC) ===")
    print(f"  0% (mammoscreen only) = {m[0]:.4f}")
    print(f"  100% (full CBIS)      = {m[i100]:.4f}   [db~{int(nimg[i100])} imgs]")
    print(f"  200% (CBIS+all InBr)  = {m[-1]:.4f}   [db~{int(nimg[-1])} imgs]")
    print(f"  delta(200-100)        = {m[-1]-m[i100]:+.4f}")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
