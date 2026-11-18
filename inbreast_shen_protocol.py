"""INbreast under Shen et al. 2019's protocol (Sci Rep 9:12495), for a like-for-like comparison:
BI-RADS 1-2 negative, 4-6 positive, BI-RADS 3 EXCLUDED; patient-level split; AUC PER IMAGE
(each view scored on its own). Shen's single models: 0.95 per image after fine-tuning a
CBIS-DDSM-trained CNN on INbreast (our regime C); 4-model ensemble 0.98.
Our scores per image = the view-agnostic head's score for that image (the per-view branch of
the method); per-breast AUC (our usual unit) is reported alongside. Regimes B (INbreast only)
and C (CBIS-DDSM + INbreast); patient-stratified 20% test, seeds 0-3 pooled.
Run: python inbreast_shen_protocol.py -> cache/inbreast_shen_protocol.json"""
import json
import os

import numpy as np

import ablation_e1 as A
import bootstrap_ci as BC
import config
import db
import inbreast_eval as IE

MCLIP, DINO, RAD, MS = A.ALL4
ROWS = [("MammoFF", A.ALL4), ("MammoScreen + our classifier", [MS]),
        ("3 native + our classifier", [MCLIP, DINO, RAD])]


TEST_FRAC = float(os.environ.get("SHEN_TEST_FRAC", "0.2"))   # 0.3 = Shen's 70/30 split


def _masks(regime, seed, split, grp, y):
    inb = np.char.startswith(grp.astype(str), "inbreast_p")
    te = np.zeros(len(grp), bool)
    te[inb] = IE.M.patient_test_mask(grp[inb], y[inb], TEST_FRAC, seed)
    cbis = ~inb & (split == "train")
    tr = (inb & ~te) | (cbis if regime == "C_cbis_plus_inbreast" else np.zeros_like(te))
    return tr, te


def main():
    conn = db.connect(); out = {"test_frac": TEST_FRAC}
    for regime, datasets in (("B_within", ["inbreast_full"]),
                             ("C_cbis_plus_inbreast", ["cbis_full", "inbreast_full"])):
        out[regime] = {}
        for name, encs in ROWS:
            ids, X, y, split, grp, view, ds, patho, abn = A.load(conn, datasets, encs, "bm", True)
            keep = patho != "BIRADS_3"                      # Shen: BI-RADS 3 excluded
            ids, X, y, split, grp, view = ids[keep], X[keep], y[keep], split[keep], grp[keep], view[keep]
            img, brs = {}, {}
            for seed in IE.SEEDS:
                tr, te = _masks(regime, seed, split, grp, y)
                H = A.heads(ids, X, y, grp, view, tr, te, seed)
                lab = dict(zip(ids.tolist(), y.tolist()))
                img.update({f"{seed}:{i}": (lab[i], s) for i, s in H["all"][1].items()})
                ted, _ = A.fuse(conn, H, ids, view, te, "default", "te")
                brs.update({f"{seed}:{c}": v for c, v in ted.items()})
            res = {}
            for unit, d in (("per_image", img), ("per_breast", brs)):
                k = sorted(d); yy = np.array([d[c][0] for c in k]); ss = np.array([d[c][1] for c in k])
                ci, n = BC.ci(yy, ss)
                res[unit] = dict(n=n, auc=ci["auc"], sens=ci["sens"], spec=ci["spec"])
            out[regime][name] = res
            print(f"{regime:22s} {name:30s} per-image AUC {res['per_image']['auc']['est']:.3f} "
                  f"[{res['per_image']['auc']['lo']:.3f}, {res['per_image']['auc']['hi']:.3f}] (n={res['per_image']['n']})  "
                  f"per-breast {res['per_breast']['auc']['est']:.3f} (n={res['per_breast']['n']})", flush=True)
    tag = "" if TEST_FRAC == 0.2 else f"_test{int(TEST_FRAC*100)}"
    json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", f"inbreast_shen_protocol{tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
