"""Preprocessing check: INbreast at native resolution, uncropped (inbreast_full, the paper's
version) vs INbreast through the RSNA pipeline (inbreast_512: mammo-crop + 512x512, like the RSNA
subset). Same test patients (paper's 20% split, seeds 0-3), paired deltas 512 minus full.
If RSNA-trained regimes gain on inbreast_512, the RSNA results were held back by the
preprocessing mismatch. Also D/E on ALL INbreast (external).
Run: python inbreast512_eval.py -> cache/inbreast512_eval.json"""
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
REGIMES = {"B_within": [], "A_cbis_to_inbreast": ["cbis_full"], "D_rsna": ["rsna_sub"],
           "E_rsna_cbis": ["rsna_sub", "cbis_full"], "F_rsna_inbreast": ["rsna_sub"],
           "G_rsna_cbis_inbreast": ["rsna_sub", "cbis_full"]}
OUT = os.path.join(config.PROJECT_DIR, "cache", "inbreast512_eval.json")


def scores(conn, encs, inb_ds, regime, extra):
    ids, X, y, split, grp, view, *_ = A.load(conn, extra + [inb_ds], encs, "bm", True)
    d = {}
    for seed in IE.SEEDS:
        tr, te = IE.masks(regime, seed, split, grp, y)
        H = A.heads(ids, X, y, grp, view, tr, te, seed)
        ted, _ = A.fuse(conn, H, ids, view, te, "default", "te")
        d.update({f"{seed}:{c}": v for c, v in ted.items()})
    ext = None
    if regime in ("A_cbis_to_inbreast", "D_rsna", "E_rsna_cbis"):      # all INbreast, external
        g = grp.astype(str); inb = np.char.startswith(g, "inbreast_p"); rsna = np.char.startswith(g, "rsna_")
        tr = rsna | (~inb & ~rsna & (split == "train"))
        H = A.heads(ids, X, y, grp, view, tr, inb, 0)
        ext, _ = A.fuse(conn, H, ids, view, inb, "default", "te")
    return d, ext


def summ(d):
    cs = sorted(d)
    ci, n = BC.ci(np.array([d[c][0] for c in cs]), np.array([d[c][1] for c in cs]), cs)
    return ci, n


def main():
    conn = db.connect(); out = {}
    for name, encs in ROWS:
        out[name] = {}
        for regime, extra in REGIMES.items():
            full, ext_full = scores(conn, encs, "inbreast_full", regime, extra)
            low, ext_low = scores(conn, encs, "inbreast_512", regime, extra)
            for tag, a, b in (("split", full, low), ("all-INbreast", ext_full, ext_low)):
                if a is None:
                    continue
                cs = sorted(set(a) & set(b))
                cf, _ = summ(a); cl, n = summ(b)
                dl, dci = A.delta_ci(a, b, cs)
                out[name][f"{regime} {tag}"] = dict(full=cf, low512=cl, delta=dl, delta_ci=dci, n=n)
                print(f"{name:30s} {regime:22s} {tag:12s} full {cf['auc']['est']:.3f}  512 {cl['auc']['est']:.3f} "
                      f"[{cl['auc']['lo']:.3f},{cl['auc']['hi']:.3f}]  diff {dl:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}]  "
                      f"512 acc {cl['acc']['est']*100:.1f} sens {cl['sens']['est']*100:.1f} spec {cl['spec']['est']*100:.1f}",
                      flush=True)
            json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
