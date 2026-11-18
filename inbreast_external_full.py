"""INbreast as a fully EXTERNAL test set (all patients, no INbreast data in training), the setting
of ExViT-Breast 2026 (CBIS-DDSM -> all INbreast) and Ribli 2018 (DDSM + private FFDM -> all INbreast).
Our label rule as in the paper (BI-RADS 1-3 vs 4-6), per breast, threshold 0.5, 5,000 bootstrap.
Regimes: A CBIS-DDSM training set; D RSNA subset; E RSNA + CBIS-DDSM.
Run: python inbreast_external_full.py -> cache/inbreast_external_full.json"""
import json
import os

import numpy as np

import ablation_e1 as A
import bootstrap_ci as BC
import config
import db

MCLIP, DINO, RAD, MS = A.ALL4
ROWS = [("MammoFF", A.ALL4), ("MammoScreen + our classifier", [MS]),
        ("3 native + our classifier", [MCLIP, DINO, RAD])]
REGIMES = {"A_cbis": ["cbis_full", "inbreast_full"], "D_rsna": ["rsna_sub", "inbreast_full"],
           "E_rsna_cbis": ["rsna_sub", "cbis_full", "inbreast_full"]}


def main():
    conn = db.connect(); out = {}
    for regime, datasets in REGIMES.items():
        out[regime] = {}
        for name, encs in ROWS:
            ids, X, y, split, grp, view, *_ = A.load(conn, datasets, encs, "bm", True)
            g = grp.astype(str)
            te = np.char.startswith(g, "inbreast_p")
            rsna = np.char.startswith(g, "rsna_")
            tr = rsna | (~te & ~rsna & (split == "train"))
            H = A.heads(ids, X, y, grp, view, tr, te, 0)
            ted, _ = A.fuse(conn, H, ids, view, te, "default", "te")
            cs = sorted(ted)
            ci, n = BC.ci(np.array([ted[c][0] for c in cs]), np.array([ted[c][1] for c in cs]))
            out[regime][name] = dict(n_breasts=n, metrics=ci)
            print(f"{regime:12s} {name:30s} AUC {ci['auc']['est']:.3f} [{ci['auc']['lo']:.3f},{ci['auc']['hi']:.3f}] "
                  f"acc {ci['acc']['est']*100:.1f} sens {ci['sens']['est']*100:.1f} spec {ci['spec']['est']*100:.1f} "
                  f"(n={n})", flush=True)
            json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", "inbreast_external_full.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
