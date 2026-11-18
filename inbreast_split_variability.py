"""How much does a SINGLE random patient-level 70/30 split move the INbreast AUC?
Shen et al. 2019's protocol (BI-RADS 3 excluded, per-image AUC, 70/30 patient split), MammoFF,
regimes B (INbreast only) and C (CBIS-DDSM training + INbreast, Shen's training setup),
repeated over N_SPLITS random splits (seeds 0..N-1). Reports the distribution of single-split
AUCs, i.e. the range a one-split study could have reported with the same method and data.
Run: python inbreast_split_variability.py -> cache/inbreast_split_variability.json"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import ablation_e1 as A
import config
import db
import miniddsm_B as M

N_SPLITS = int(os.environ.get("N_SPLITS", "100"))
TEST_FRAC = 0.3


def main():
    conn = db.connect(); out = {"n_splits": N_SPLITS, "test_frac": TEST_FRAC}
    for regime, datasets in (("B_within", ["inbreast_full"]),
                             ("C_cbis_plus_inbreast", ["cbis_full", "inbreast_full"])):
        ids, X, y, split, grp, view, ds, patho, abn = A.load(conn, datasets, A.ALL4, "bm", True)
        keep = patho != "BIRADS_3"
        ids, X, y, split, grp, view = ids[keep], X[keep], y[keep], split[keep], grp[keep], view[keep]
        inb = np.char.startswith(grp.astype(str), "inbreast_p")
        cbis = ~inb & (split == "train")
        img_auc, br_auc = [], []
        for seed in range(N_SPLITS):
            te = np.zeros(len(grp), bool)
            te[inb] = M.patient_test_mask(grp[inb], y[inb], TEST_FRAC, seed)
            tr = (inb & ~te) | (cbis if regime.startswith("C") else np.zeros_like(te))
            H = A.heads(ids, X, y, grp, view, tr, te, seed)
            lab = dict(zip(ids.tolist(), y.tolist()))
            sc = H["all"][1]
            img_auc.append(roc_auc_score([lab[i] for i in sc], list(sc.values())))
            ted, _ = A.fuse(conn, H, ids, view, te, "default", "te")
            br_auc.append(roc_auc_score([v[0] for v in ted.values()], [v[1] for v in ted.values()]))
            if seed % 20 == 19:
                print(f"[{regime}] {seed + 1} splits", flush=True)
        r = {}
        for unit, a in (("per_image", np.array(img_auc)), ("per_breast", np.array(br_auc))):
            r[unit] = dict(median=float(np.median(a)), mean=float(a.mean()), min=float(a.min()),
                           max=float(a.max()), p5=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
                           frac_ge_095=float((a >= 0.95).mean()), all=a.round(4).tolist())
        out[regime] = r
        p = r["per_image"]
        print(f"{regime:22s} per-image single-split AUC: median {p['median']:.3f}, 5-95% "
              f"[{p['p5']:.3f}, {p['p95']:.3f}], range [{p['min']:.3f}, {p['max']:.3f}], "
              f">=0.95 in {p['frac_ge_095']*100:.0f}% of splits", flush=True)
    json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", "inbreast_split_variability.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
