"""Every combination of training sources, tested on CBIS-DDSM and on Mini-DDSM (test sets unchanged).
Mirrors the INbreast regimes A-G from the other direction.

Training sources
  own      : the test dataset's own training data (CBIS-DDSM official training set; or the
             de-duplicated Mini-DDSM training patients, 20% patient split, seeds 0-3)
  INbreast : ALL INbreast patients (label BI-RADS 1-3 vs 4-6; INbreast is not tested here)
  RSNA     : RSNA subset, training only (MammoScreen saw an unknown 80% of RSNA, never tested)
  RSNA-bx  : (CBIS-DDSM only) only biopsied RSNA images, i.e. cancer or biopsy-proven benign,
             the closest RSNA match to the CBIS-DDSM benign-vs-malignant label
Mini-DDSM is never used to train for CBIS-DDSM and vice versa: both are drawn from DDSM, so the
same films appear in both.

Model choice uses no test data: 'oof_own' is the cross-validated AUC on the test dataset's OWN
training cases only, available whenever 'own' is a source.
Run: python rsna_train_eval.py -> cache/rsna_train_eval.json"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import ablation_e1 as A
import config
import db
import miniddsm_B as M

MCLIP, DINO, RAD, MS = A.ALL4
ROWS = [("MammoFF", A.ALL4), ("MammoScreen + our classifier", [MS]),
        ("3 native + our classifier", [MCLIP, DINO, RAD])]
RSNA_CSV = r"F:\Downloads\rsna\train.csv"
POS = ["CANCER", "MALIGNANT"]
OUT = os.path.join(config.PROJECT_DIR, "cache", "rsna_train_eval.json")


def biopsy_flags(conn, ids):
    d = pd.read_csv(RSNA_CSV)
    bio = {f"{p}_{i}.jpg": int(b) for p, i, b in zip(d.patient_id, d.image_id, d.biopsy)}
    fn = {int(r[0]): r[1] for r in conn.execute("SELECT id,file_name FROM samples WHERE dataset='rsna_sub'")}
    return np.array([bio.get(fn.get(int(i), ""), 0) for i in ids])


def auc_of(d):
    k = sorted(d)
    return roc_auc_score([d[c][0] for c in k], [d[c][1] for c in k])


def regimes(sources):
    """All non-empty combinations; RSNA and RSNA-bx are alternatives, never together."""
    out = []
    for r in range(1, len(sources) + 1):
        for combo in itertools.combinations(sources, r):
            if "RSNA" in combo and "RSNA-bx" in combo:
                continue
            out.append(combo)
    return out


def load(conn, tag, encs):
    if tag == "cbis":
        datasets = ["cbis_full", "inbreast_full", "rsna_sub"]
    else:
        datasets = ["miniddsm", "miniddsm_normal", "inbreast_full", "rsna_sub"]
    ids, X, y, split, grp, view, ds, patho, abn = A.load(conn, datasets, encs, "bm", True)
    g = grp.astype(str)
    inb = np.char.startswith(g, "inbreast_p"); rsna = np.char.startswith(g, "rsna_"); own = ~inb & ~rsna
    if tag == "mini":                       # normal vs cancer on Mini-DDSM rows; others keep their label
        keep = ~own | np.isin(patho, ["NORMAL"] + POS)
        y = y.copy(); y[own] = np.isin(patho[own], POS).astype(int)
        ids, X, y, split, grp, view = ids[keep], X[keep], y[keep], split[keep], grp[keep], view[keep]
        inb, rsna, own = inb[keep], rsna[keep], own[keep]
    return ids, X, y, split, grp, view, inb, rsna, own


def run(conn, tag, name, encs):
    ids, X, y, split, grp, view, inb, rsna, own = load(conn, tag, encs)
    bio = biopsy_flags(conn, ids)
    srcs = ["own", "INbreast", "RSNA"] + (["RSNA-bx"] if tag == "cbis" else [])
    per = {}
    for seed in A.SEEDS:
        if tag == "cbis":
            own_tr = own & (split == "train"); te = own & (split == "test")
        else:
            te = np.zeros(len(grp), bool)
            te[own] = M.patient_test_mask(grp[own], y[own], 0.2, seed)
            own_tr = own & ~te
        pool = {"own": own_tr, "INbreast": inb, "RSNA": rsna, "RSNA-bx": rsna & ((y == 1) | (bio == 1))}
        for combo in regimes(srcs):
            tr = np.logical_or.reduce([pool[s] for s in combo])
            H = A.heads(ids, X, y, grp, view, tr, te, seed)
            ted, _ = A.fuse(conn, H, ids, view, te, "default", "te")
            oof = np.nan
            if "own" in combo:
                trd, _ = A.fuse(conn, H, ids, view, own_tr, "default", "tr")
                oof = auc_of(trd)
            key = " + ".join(combo)
            p = per.setdefault(key, dict(te=[], oof=[], n_train=int(tr.sum())))
            p["te"].append(ted); p["oof"].append(oof)
        print(f"[{tag}] {name}: seed {seed} done", flush=True)
    return per


def summarise(tag, name, per):
    ref = per["own"]; res = {}
    for regime, p in per.items():
        aucs = [auc_of(d) for d in p["te"]]
        seeds = [0] if tag == "cbis" else range(len(p["te"]))          # CBIS test set is fixed
        rd, vd = {}, {}
        for k in seeds:
            for c, v in p["te"][k].items():
                rd[f"{k}:{c}"] = ref["te"][k][c]; vd[f"{k}:{c}"] = v
        dl, dci = A.delta_ci(rd, vd, list(rd)) if regime != "own" else (0.0, [0.0, 0.0])
        oof = float(np.nanmean(p["oof"])) if not np.all(np.isnan(p["oof"])) else None
        res[regime] = dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)), oof_own=oof,
                           delta_vs_own=dl, delta_ci=dci, n_train_images=p["n_train"])
        print(f"[{tag}] {name:30s} {regime:28s} AUC {np.mean(aucs):.3f}  vs own {dl:+.3f} "
              f"[{dci[0]:+.3f},{dci[1]:+.3f}]  own-OOF {'-' if oof is None else f'{oof:.3f}'}  "
              f"(train imgs {p['n_train']})", flush=True)
    return res



def main():
    conn = db.connect(); out = {}
    for tag in ("cbis", "mini"):
        out[tag] = {}
        for name, encs in ROWS:
            out[tag][name] = summarise(tag, name, run(conn, tag, name, encs))
            json.dump(out, open(OUT, "w"), indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
