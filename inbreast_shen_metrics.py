"""Our top 3 on INbreast under Shen et al. 2019's rules, with full metrics.
Shen's rules: BI-RADS 1-2 negative, 4-6 positive, BI-RADS 3 EXCLUDED; one 70/30 patient split;
AUC per IMAGE; Shen report AUC (0.95 single model, 0.98 ensemble) and, for the ensemble only,
sensitivity 86.7% / specificity 96.1% (an ROC point; no accuracy reported).
Rows: MammoFF; MammoScreen + our classifier; MammoScreen as released (own score, one per BREAST,
so it has no per-image value). Regimes B (INbreast only) and C (CBIS-DDSM + INbreast, Shen's
training data). Metrics: AUC, accuracy/sens/spec at the fixed 0.5 cutoff, and specificity and
accuracy at sensitivity >= 86.7% (Shen's operating point; like Shen, read off the test ROC).
Reported over N_SPLITS random 70/30 splits: median and 5-95% band of the single-split values.
Run: python inbreast_shen_metrics.py -> cache/inbreast_shen_metrics.json"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import ablation_e1 as A
import config
import db
import miniddsm_B as M

MCLIP, DINO, RAD, MS = A.ALL4
N_SPLITS = int(os.environ.get("N_SPLITS", "100"))
TEST_FRAC = 0.3
SHEN_SENS = 0.867


def metrics(y, s):
    y = np.asarray(y); s = np.asarray(s); p = (s >= 0.5).astype(int)
    fpr, tpr, thr = roc_curve(y, s)
    k = int(np.argmax(tpr >= SHEN_SENS))                   # first ROC point reaching Shen's sensitivity
    sens_k, spec_k = tpr[k], 1 - fpr[k]
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    return dict(auc=roc_auc_score(y, s), acc=(p == y).mean(), sens=(p[y == 1] == 1).mean(),
                spec=(p[y == 0] == 0).mean(), spec_at_shen_sens=spec_k,
                acc_at_shen_sens=(sens_k * npos + spec_k * nneg) / (npos + nneg))


def main():
    conn = db.connect()
    z = np.load(os.path.join(config.PROJECT_DIR, "cache", "mammoscreen_inbreast_zeroshot.npz"), allow_pickle=True)
    released = {str(c): float(s) for c, s in zip(z["cids"], z["s"])}
    out = {"n_splits": N_SPLITS, "test_frac": TEST_FRAC, "shen_sens": SHEN_SENS}
    for regime, datasets in (("B_within", ["inbreast_full"]),
                             ("C_cbis_plus_inbreast", ["cbis_full", "inbreast_full"])):
        res = {}
        data = {}
        for name, encs in (("MammoFF", A.ALL4), ("MammoScreen + our classifier", [MS])):
            ids, X, y, split, grp, view, ds, patho, abn = A.load(conn, datasets, encs, "bm", True)
            k = patho != "BIRADS_3"
            data[name] = (ids[k], X[k], y[k], split[k], grp[k], view[k])
        for seed in range(N_SPLITS):
            for name, (ids, X, y, split, grp, view) in data.items():
                inb = np.char.startswith(grp.astype(str), "inbreast_p")
                te = np.zeros(len(grp), bool)
                te[inb] = M.patient_test_mask(grp[inb], y[inb], TEST_FRAC, seed)
                tr = (inb & ~te) | ((~inb & (split == "train")) if regime.startswith("C") else np.zeros_like(te))
                H = A.heads(ids, X, y, grp, view, tr, te, seed)
                lab = dict(zip(ids.tolist(), y.tolist())); sc = H["all"][1]
                img = metrics([lab[i] for i in sc], list(sc.values()))
                ted, _ = A.fuse(conn, H, ids, view, te, "default", "te")
                br = metrics([v[0] for v in ted.values()], [v[1] for v in ted.values()])
                res.setdefault(name, {"per_image": [], "per_breast": []})
                res[name]["per_image"].append(img); res[name]["per_breast"].append(br)
                if name == "MammoFF":                        # released score on the same test breasts
                    cs = [c for c in ted if c in released]
                    res.setdefault("MammoScreen as released", {"per_breast": []})["per_breast"].append(
                        metrics([ted[c][0] for c in cs], [released[c] for c in cs]))
            if seed % 20 == 19:
                print(f"[{regime}] {seed + 1} splits", flush=True)
        summ = {}
        for name, units in res.items():
            summ[name] = {}
            for unit, lst in units.items():
                summ[name][unit] = {m: dict(median=float(np.median([d[m] for d in lst])),
                                            p5=float(np.percentile([d[m] for d in lst], 5)),
                                            p95=float(np.percentile([d[m] for d in lst], 95)))
                                    for m in lst[0]}
                s = summ[name][unit]
                print(f"{regime:20s} {name:30s} {unit:10s} AUC {s['auc']['median']:.3f} "
                      f"[{s['auc']['p5']:.3f},{s['auc']['p95']:.3f}]  acc@0.5 {s['acc']['median']*100:.1f} "
                      f"sens {s['sens']['median']*100:.1f} spec {s['spec']['median']*100:.1f} | "
                      f"@sens86.7: spec {s['spec_at_shen_sens']['median']*100:.1f} "
                      f"acc {s['acc_at_shen_sens']['median']*100:.1f}", flush=True)
        out[regime] = summ
        json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", "inbreast_shen_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
