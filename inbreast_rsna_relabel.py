"""Does the RSNA LABEL contradict the INbreast label?
INbreast positive = BI-RADS 4-6 (suspicious, includes lesions later found benign). RSNA label =
biopsy-proven cancer, so RSNA images that were biopsied but benign (the RSNA counterpart of a
BI-RADS 4 finding) are NEGATIVE in training but would be POSITIVE under the INbreast rule.
RSNA label variants (training only):
  cancer      : as used so far (cancer = 1)
  suspicious  : cancer OR biopsied = 1  (closest to BI-RADS 4-6)
  drop_bx     : biopsied-benign images removed from training
Regimes D (RSNA), E (RSNA + CBIS), F (RSNA + INbreast), G (all three); INbreast test = the paper's
20% patient split, seeds 0-3 pooled (same 171 breasts), plus D/E on ALL INbreast (external).
Run: python inbreast_rsna_relabel.py -> cache/inbreast_rsna_relabel.json"""
import json
import os

import numpy as np

import ablation_e1 as A
import bootstrap_ci as BC
import config
import db
import inbreast_eval as IE
from rsna_train_eval import biopsy_flags

MCLIP, DINO, RAD, MS = A.ALL4
ROWS = [("MammoFF", A.ALL4), ("MammoScreen + our classifier", [MS]),
        ("3 native + our classifier", [MCLIP, DINO, RAD])]
VARIANTS = ["cancer", "suspicious", "drop_bx"]
OUT = os.path.join(config.PROJECT_DIR, "cache", "inbreast_rsna_relabel.json")


def main():
    conn = db.connect(); out = {}
    for name, encs in ROWS:
        ids, X, y0, split, grp, view, *_ = A.load(conn, ["rsna_sub", "cbis_full", "inbreast_full"], encs, "bm", True)
        g = grp.astype(str)
        rsna = np.char.startswith(g, "rsna_"); inb = np.char.startswith(g, "inbreast_p")
        bxb = rsna & (biopsy_flags(conn, ids) == 1) & (y0 == 0)
        print(f"{name}: RSNA biopsied-benign images {int(bxb.sum())}", flush=True)
        te_all = {}
        for var in VARIANTS:
            y = y0.copy(); drop = np.zeros_like(rsna)
            if var == "suspicious":
                y[bxb] = 1
            elif var == "drop_bx":
                drop = bxb
            for regime in ("D_rsna", "E_rsna_cbis", "F_rsna_inbreast", "G_rsna_cbis_inbreast"):
                key = f"{regime} | {var}"; te_all[key] = {}
                for seed in IE.SEEDS:
                    tr, te = IE.masks(regime, seed, split, grp, y)
                    H = A.heads(ids, X, y, grp, view, tr & ~drop, te, seed)
                    ted, _ = A.fuse(conn, H, ids, view, te, "default", "te")
                    te_all[key].update({f"{seed}:{c}": v for c, v in ted.items()})
            for regime in ("D_rsna", "E_rsna_cbis"):        # INbreast fully external
                tr = rsna | ((~rsna & ~inb & (split == "train")) if regime == "E_rsna_cbis" else np.zeros_like(rsna))
                H = A.heads(ids, X, y, grp, view, tr & ~drop, inb, 0)
                ted, _ = A.fuse(conn, H, ids, view, inb, "default", "te")
                te_all[f"{regime} ALL-INbreast | {var}"] = ted
        res = {}
        for key, d in te_all.items():
            cs = sorted(d)
            ci, n = BC.ci(np.array([d[c][0] for c in cs]), np.array([d[c][1] for c in cs]), cs)
            ref = te_all[key.rsplit(" | ", 1)[0] + " | cancer"]
            dl, dci = (0.0, [0.0, 0.0]) if key.endswith("| cancer") else A.delta_ci(ref, d, cs)
            res[key] = dict(n=n, metrics=ci, delta_vs_cancer_label=dl, delta_ci=dci)
            print(f"{name:30s} {key:42s} AUC {ci['auc']['est']:.3f} [{ci['auc']['lo']:.3f},{ci['auc']['hi']:.3f}] "
                  f"vs cancer-label {dl:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}] acc {ci['acc']['est']*100:.1f} "
                  f"sens {ci['sens']['est']*100:.1f} spec {ci['spec']['est']*100:.1f}", flush=True)
        out[name] = res
        json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
