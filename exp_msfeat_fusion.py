"""User-requested experiment: EARLY (feature-level) fusion instead of LATE (score) fusion.

Idea (user's drawing): instead of taking MammoScreen's final scalar cancer score M and
percentile-fusing it with the probe score R, take MammoScreen's PENULTIMATE feature
vector (the 3x EfficientNetV2-S pooled features, encoder `mammoscreen_feat`) and feed it
into the MVLP as a 4th vector alongside the 3 frozen encoders. Retrain the probe on the
4-vector concatenation and use its output directly -- no percentile fusion, no weight w.

For each dataset this prints, on the SAME paired-case test subset and protocol used for
the paper's headline numbers:
  BASE3 probe-alone R        (sanity: must reproduce ~0.854 CBIS / ~0.973 Mini-DDSM)
  BASE3 late-fusion (w=0.6)  (sanity: must reproduce ~0.896 CBIS / ~0.961 Mini-DDSM)
  BASE4 probe-alone R        (the new early-fusion number -- the thing being tested)
  BASE4 late-fusion (w=0.6)  (for completeness: early + late together)

BASE4 = BASE3 + mammoscreen_feat. Everything else (CV head, case fusion, honest train
threshold, 4 seeds, paired subset) is identical to miniddsm_B / paper_numbers.
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import config, db
import miniddsm_B as M
from data_scaling_extend import head_case_scores, align_B, _l2
from data_scaling import _pct
from mil import _best_threshold, _metrics

BASE3 = ["mammo_clip_b5", "dinov2_vitl14_r518_noclahe", "rad_dino_noclahe"]
MSFEAT = "mammoscreen_feat"
W = 0.6


def load_enc(conn, datasets, encoders):
    """miniddsm_B.load_base, but with an explicit encoder list (so we can add a 4th)."""
    ids0, mats = None, []
    for e in encoders:
        ids, v, _ = db.load_embeddings(conn, e, dataset=datasets)
        if ids0 is None:
            ids0 = ids
        else:
            assert np.array_equal(ids, ids0), f"embedding misalignment for {e}"
        mats.append(_l2(v))
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


def cbis_ms():
    full = {}
    for sp in ("train", "test"):
        z = np.load(os.path.join(config.PROJECT_DIR, "cache", f"mammoscreen_{sp}.npz"),
                    allow_pickle=True)
        full.update({str(c): float(s) for c, s in zip(z["cids"], z["s"])})
    return full


def scores_one_seed(conn, ids, X, y, grp, view, tr, te, ms_map, seed):
    """Return (yte, Rte, Mte): case-level test labels, probe score, mammoscreen score."""
    rcf_tr, rcf_te = head_case_scores(conn, ids, X, y, grp, view, tr, te, M.C_HEAD, seed)
    ytr, Rtr, Mtr, yte, Rte, Mte = align_B(rcf_tr, rcf_te, ms_map, ms_map)
    return ytr, Rtr, Mtr, yte, Rte, Mte


def metrics_probe(ytr, Rtr, yte, Rte):
    """AUC + honest-threshold acc/bal for a probe score alone (percentile-normed)."""
    Rp, Rpe = _pct(Rtr, Rtr), _pct(Rte, Rtr)
    thr = _best_threshold(ytr, Rp)
    m = _metrics(yte, Rpe, thr)
    return roc_auc_score(yte, Rte), m["accuracy"], m["bal_acc"]


def metrics_fused(ytr, Rtr, Mtr, yte, Rte, Mte):
    Rp, Mp = _pct(Rtr, Rtr), _pct(Mtr, Mtr)
    Rpe, Mpe = _pct(Rte, Rtr), _pct(Mte, Mtr)
    ftr, fte = W * Rp + (1 - W) * Mp, W * Rpe + (1 - W) * Mpe
    thr = _best_threshold(ytr, ftr)
    m = _metrics(yte, fte, thr)
    return roc_auc_score(yte, fte), m["accuracy"], m["bal_acc"]


def run_dataset(conn, label, X3, X4, y, grp, view, ids, seeds, split_fn, ms_map):
    agg = {k: [] for k in ("R3", "F3", "R4", "F4")}
    for seed in seeds:
        tr, te = split_fn(seed)
        # BASE3
        ytr, Rtr, Mtr, yte, Rte, Mte = scores_one_seed(conn, ids, X3, y, grp, view, tr, te, ms_map, seed)
        agg["R3"].append(metrics_probe(ytr, Rtr, yte, Rte))
        agg["F3"].append(metrics_fused(ytr, Rtr, Mtr, yte, Rte, Mte))
        # BASE4 (same splits/cases; probe now sees mammoscreen features too)
        ytr4, Rtr4, Mtr4, yte4, Rte4, Mte4 = scores_one_seed(conn, ids, X4, y, grp, view, tr, te, ms_map, seed)
        agg["R4"].append(metrics_probe(ytr4, Rtr4, yte4, Rte4))
        agg["F4"].append(metrics_fused(ytr4, Rtr4, Mtr4, yte4, Rte4, Mte4))

    def show(key, name):
        a = np.array(agg[key])
        mu, sd = a.mean(0), a.std(0)
        print(f"    {name:34s} AUC={mu[0]:.4f}±{sd[0]:.4f}  "
              f"acc={mu[1]:.4f}  bal={mu[2]:.4f}")

    print(f"\n=== {label} ===  (X3 dim={X3.shape[1]}, X4 dim={X4.shape[1]}, "
          f"{len(seeds)} seeds)")
    show("R3", "BASE3 probe-alone   (~paper R)")
    show("F3", "BASE3 late-fusion   (~paper 0.896/0.961)")
    show("R4", "BASE4 probe-alone   [EARLY FUSION]")
    show("F4", "BASE4 early+late    (probe4 + score)")


def main():
    conn = db.connect()

    # ---------- CBIS-DDSM benign/malignant (official split) ----------
    ids, X3, y, split, grp, view, ds, patho, abn = load_enc(conn, ["cbis_full"], BASE3)
    _, X4, *_ = load_enc(conn, ["cbis_full"], BASE3 + [MSFEAT])
    ms_c = cbis_ms()
    run_dataset(conn, "CBIS-DDSM benign/malignant (official split)",
                X3, X4, y, grp, view, ids, [0, 1, 2, 3],
                lambda seed: (split == "train", split == "test"), ms_c)

    # ---------- Mini-DDSM normal/cancer (20% patient-disjoint) ----------
    dsets = ["miniddsm", "miniddsm_normal"]
    ms_ids = db.load_embeddings(conn, MSFEAT, dataset=dsets)[0]
    base_ids = db.load_embeddings(conn, BASE3[0], dataset=dsets)[0]
    n_ms = 0 if ms_ids is None else len(ms_ids)
    n_base = 0 if base_ids is None else len(base_ids)
    if n_ms < n_base:
        print(f"\n[Mini-DDSM] mammoscreen_feat coverage {n_ms}/{n_base} -- still caching, "
              "skipping. Re-run when embed_all finishes.")
        return
    raw3 = load_enc(conn, ["miniddsm", "miniddsm_normal"], BASE3)
    raw4 = load_enc(conn, ["miniddsm", "miniddsm_normal"], BASE3 + [MSFEAT])
    ids, X3, y, split, grp, view, ds, patho, abn = M.apply_task("nc", *raw3)
    # apply_task filters rows; must apply the SAME mask to X4. Re-derive via patho on raw4.
    ids4, X4full, y4, split4, grp4, view4, ds4, patho4, abn4 = raw4
    POS = ["CANCER", "MALIGNANT"]
    m = np.isin(patho4, ["NORMAL"] + POS)
    X4 = X4full[m]
    assert np.array_equal(ids, ids4[m]), "BASE3/BASE4 row mismatch after nc filter"
    ms_m = M.load_ms(["miniddsm", "miniddsm_normal"])
    run_dataset(conn, "Mini-DDSM normal/cancer (20% patient-disjoint)",
                X3, X4, y, grp, view, ids, [0, 1, 2, 3],
                lambda seed: (~M.patient_test_mask(grp, y, 0.2, seed),
                              M.patient_test_mask(grp, y, 0.2, seed)), ms_m)


if __name__ == "__main__":
    main()
