"""INbreast (full-field DIGITAL mammography) under the paper's exact method, three training regimes.

Task: BI-RADS 1-3 = 0, BI-RADS 4-6 = 1 (radiologist assessment; INbreast has no pathology for
every case). Patients: exams rebuilt by inbreast_groups.py (116, anonymised IDs). Breast = case.
Test cases: the SAME INbreast patients in every regime (patient-stratified 20%, seeds 0-3),
so regimes are compared with paired bootstrap CIs.

  B  within : fit on INbreast training patients            -> test INbreast test patients
  A  film   : fit on CBIS-DDSM official training set only  -> same INbreast test patients
  C  both   : fit on CBIS-DDSM training + INbreast training -> same INbreast test patients

Everything else as in the paper: frozen features, L2 per extractor, three logistic heads
(C=0.5, balanced), four branches averaged per breast, threshold 0.5, 5,000 bootstrap resamples.
Run: python inbreast_eval.py  ->  cache/inbreast_eval.json
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import ablation_e1 as A
import bootstrap_ci as BC
import config
import db
import miniddsm_B as M

MCLIP, DINO, RAD, MS = A.ALL4
FM = "mammo_fm"
SEEDS = [0, 1, 2, 3]
OUT = os.path.join(config.PROJECT_DIR, "cache", "inbreast_eval.json")   # RSNA run: inbreast_eval_rsna.json

CORE = [("All four (MammoFF)", A.ALL4, True, "default"),
        ("Without MammoScreen (three native)", [MCLIP, DINO, RAD], True, "default"),
        ("MammoScreen alone", [MS], True, "default"),
        ("Mammo-CLIP alone", [MCLIP], True, "default"),
        ("Mammo-FM replaces MammoScreen", [MCLIP, DINO, RAD, FM], True, "default"),
        ("Mammo-FM alone", [FM], True, "default")]
EXTRA_B = [c for c in A.default_configs()[1:] if c[0] not in ("MammoScreen alone", "Mammo-CLIP alone",
                                                              "Without MammoScreen")]
EXTRA_B += [("Three native without Mammo-CLIP", [DINO, RAD], True, "default"),
            ("Three native without DINOv2", [MCLIP, RAD], True, "default"),
            ("Three native without RAD-DINO", [MCLIP, DINO], True, "default"),
            ("Mammo-FM replaces MammoScreen, no RAD-DINO", [MCLIP, DINO, FM], True, "default")]
REGIMES = {"B_within": (["inbreast_full"], CORE + EXTRA_B),
           "A_cbis_to_inbreast": (["cbis_full", "inbreast_full"], CORE),
           "C_cbis_plus_inbreast": (["cbis_full", "inbreast_full"], CORE),
           # RSNA subset (prep_rsna.py) is TRAINING-ONLY: MammoScreen saw an unknown 80% of RSNA
           "D_rsna": (["rsna_sub", "inbreast_full"], CORE),
           "E_rsna_cbis": (["rsna_sub", "cbis_full", "inbreast_full"], CORE),
           "F_rsna_inbreast": (["rsna_sub", "inbreast_full"], CORE),
           "G_rsna_cbis_inbreast": (["rsna_sub", "cbis_full", "inbreast_full"], CORE)}


def masks(regime, seed, split, grp, y):
    inb = np.char.startswith(grp.astype(str), "inbreast_p")
    te = np.zeros(len(grp), bool)
    te[inb] = M.patient_test_mask(grp[inb], y[inb], 0.2, seed)
    g = grp.astype(str)
    rsna = np.char.startswith(g, "rsna_")
    cbis = ~inb & ~rsna & (split == "train")          # CBIS-DDSM official training set
    inb_tr = inb & ~te
    use = {"B_within": [inb_tr], "A_cbis_to_inbreast": [cbis], "C_cbis_plus_inbreast": [cbis, inb_tr],
           "D_rsna": [rsna], "E_rsna_cbis": [rsna, cbis], "F_rsna_inbreast": [rsna, inb_tr],
           "G_rsna_cbis_inbreast": [rsna, cbis, inb_tr]}[regime]
    tr = np.logical_or.reduce(use)
    return tr, te


def run_regime(conn, regime, datasets, configs):
    groups = {}
    for name, encs, l2, var in configs:
        groups.setdefault((tuple(encs), l2), []).append((name, var))
    te_all = {n: {} for n, *_ in configs}; tr_auc = {n: [] for n, *_ in configs}
    for (encs, l2), members in groups.items():
        ids, X, y, split, grp, view, *_ = A.load(conn, datasets, list(encs), "bm", l2)
        for seed in SEEDS:
            tr, te = masks(regime, seed, split, grp, y)
            H = A.heads(ids, X, y, grp, view, tr, te, seed)
            for name, var in members:
                trd, _ = A.fuse(conn, H, ids, view, tr, var, "tr")
                ted, _ = A.fuse(conn, H, ids, view, te, var, "te")
                te_all[name].update({f"{seed}:{c}": v for c, v in ted.items()})
                cs = sorted(trd); tr_auc[name].append(roc_auc_score([trd[c][0] for c in cs], [trd[c][1] for c in cs]))
        print(f"[{regime}] {len(encs)} extractor(s): done", flush=True)
    return te_all, {n: float(np.mean(v)) for n, v in tr_auc.items()}


def summarise(te_all, ref_name, tr_auc):
    ref = te_all[ref_name]; res = {}
    for name, d in te_all.items():
        cases = sorted(d)
        y = np.array([d[c][0] for c in cases]); s = np.array([d[c][1] for c in cases])
        ci, n = BC.ci(y, s, cases)          # case-level: seeds' test partitions overlap
        shared = sorted(set(d) & set(ref))
        dl, dci = A.delta_ci(ref, d, shared) if name != ref_name else (0.0, [0.0, 0.0])
        res[name] = dict(n_pooled=n, metrics=ci, train_oof_auc=tr_auc[name], delta=dl, delta_ci=dci)
        print(f"   {name:44s} AUC {ci['auc']['est']:.3f} [{ci['auc']['lo']:.3f},{ci['auc']['hi']:.3f}] "
              f"diff {dl:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}] acc {ci['acc']['est']*100:.1f} "
              f"sens {ci['sens']['est']*100:.1f} spec {ci['spec']['est']*100:.1f} trOOF {tr_auc[name]:.3f}",
              flush=True)
    return res


def main():
    conn = db.connect(); out = {}; raw = {}
    import sys
    only = [a for a in sys.argv[1:] if a in REGIMES]   # e.g. python inbreast_eval.py B_within D_rsna ...
    for regime, (datasets, configs) in REGIMES.items():
        if only and regime not in only:
            continue
        te_all, tr_auc = run_regime(conn, regime, datasets, configs)
        raw[regime] = te_all
        print(f"== {regime}")
        out[regime] = summarise(te_all, "All four (MammoFF)", tr_auc)
    out["cross_regime"] = {}
    for name in [c[0] for c in CORE]:
        for r in [k for k in raw if k != "B_within"]:
            shared = sorted(set(raw[r][name]) & set(raw["B_within"][name]))
            dl, dci = A.delta_ci(raw["B_within"][name], raw[r][name], shared)
            out["cross_regime"][f"{name} | {r} minus B_within"] = dict(delta=dl, delta_ci=dci, n=len(shared))
            print(f"   {name:40s} {r:22s} vs within: {dl:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}]")
    path = OUT if not only else OUT.replace(".json", "_" + "_".join(only) + ".json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
