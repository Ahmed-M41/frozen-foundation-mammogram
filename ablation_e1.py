"""E1 ablation (reviewer R2.2, R2.3, R2.7): what each part of MammoFF contributes.

Identical protocol to the headline (base4_numbers.py): same frozen features, same
patient-grouped 5-fold heads (C=0.5, balanced), same four seeds, same case fusion, fixed
0.5 threshold. Only the piece under test changes:

  encoders   : each extractor alone; each one left out; the three encoders; all four
  heads      : view-agnostic head only; CC/MLO specialist heads only; all three (default)
  views      : CC view only; MLO view only; both (default)      (scored on paired cases)
  norm       : per-extractor L2 normalisation removed

For every variant: AUC (mean over seeds), balanced accuracy at 0.5, training out-of-fold
AUC, and a paired bootstrap 95% CI on the AUC difference against the full method on the
same cases (CBIS-DDSM: representative seed 0, its test set is fixed; Mini-DDSM: the four
seeds' test cases pooled), 5,000 resamples.

Run: python ablation_e1.py  ->  cache/ablation_e1.json (+ progress log on stdout)
"""
import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import config
import db
import evaluate
import miniddsm_B as M
import multiview_probe as MV
from exp_msfeat_fusion import load_enc

ENC = {"Mammo-CLIP": "mammo_clip_b5", "DINOv2": "dinov2_vitl14_r518_noclahe",
       "RAD-DINO": "rad_dino_noclahe", "MammoScreen": "mammoscreen_feat"}
ALL4 = list(ENC.values())
SEEDS = [0, 1, 2, 3]
THR = config.DECISION_THRESHOLD
B = 5000
OUT = os.path.join(config.PROJECT_DIR, "cache", "ablation_e1.json")


def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def load(conn, datasets, encoders, task, l2=True):
    raw = load_enc(conn, datasets, encoders)
    ids, X, y, split, grp, view, ds, patho, abn = raw
    if not l2:                                    # rebuild X without per-extractor L2
        mats = []
        for e in encoders:
            eids, v, _ = db.load_embeddings(conn, e, dataset=datasets)
            pos = {int(i): k for k, i in enumerate(eids)}
            mats.append(v[[pos[int(i)] for i in ids]])
        X = np.concatenate(mats, 1).astype(np.float32)
    if task == "nc":
        return M.apply_task("nc", ids, X, y, split, grp, view, ds, patho, abn)
    return ids, X, y, split, grp, view, ds, patho, abn


def heads(ids, X, y, grp, view, tr, te, seed):
    """Fit the three heads once; return per-image OOF/test score maps for each head."""
    cc, mlo = view == "CC", view == "MLO"
    out = {}
    for name, m in (("all", np.ones_like(cc)), ("cc", cc), ("mlo", mlo)):
        oof, tes = MV.head_cv(X[tr & m], y[tr & m], grp[tr & m], X[te & m], M.C_HEAD, seed=seed)
        out[name] = ({int(i): float(s) for i, s in zip(ids[tr & m], oof)},
                     {int(i): float(s) for i, s in zip(ids[te & m], tes)})
    return out


BRANCHES = {   # variant -> list of (head, view) branches averaged per case
    "default":    [("all", "CC"), ("all", "MLO"), ("cc", "CC"), ("mlo", "MLO")],
    "general":    [("all", "CC"), ("all", "MLO")],
    "specialist": [("cc", "CC"), ("mlo", "MLO")],
    "cc_only":    [("all", "CC"), ("cc", "CC")],
    "mlo_only":   [("all", "MLO"), ("mlo", "MLO")],
}


def fuse(conn, H, ids, view, mask, variant, which):
    k = 0 if which == "tr" else 1
    br = []
    for head, v in BRANCHES[variant]:
        sel = ids[mask & (view == v)]
        mp = H[head][k]
        sel = np.array([i for i in sel if int(i) in mp])
        br.append(dict(ids=sel, score_pos=np.array([mp[int(i)] for i in sel])))
    cf = evaluate.case_fuse(conn, br, method="mean")
    return {str(c): (int(t), float(s)) for c, t, s in zip(cf["cids"], cf["y_true"], cf["y_score"])}, \
        {str(cf["cids"][i]) for i in cf["paired_idx"]}


def _metrics(d, cases):
    y = np.array([d[c][0] for c in cases]); s = np.array([d[c][1] for c in cases])
    p = (s >= THR).astype(int)
    sens = (p[y == 1] == 1).mean(); spec = (p[y == 0] == 0).mean()
    return roc_auc_score(y, s), (sens + spec) / 2


def delta_ci(ref, var, cases):
    """Paired AUC difference var - ref with a case-level bootstrap 95% CI. Keys pooled over
    seeds look like '<seed>:<case>'; one case can recur across seeds' overlapping test
    partitions, so the resampling unit is the case and all its evaluations enter together
    (identical to plain resampling when every case occurs once, e.g. CBIS-DDSM seed 0)."""
    cases = sorted(cases)
    y = np.array([ref[c][0] for c in cases])
    a = np.array([ref[c][1] for c in cases]); b = np.array([var[c][1] for c in cases])
    rng = np.random.default_rng(0)
    grp = [str(c).split(":", 1)[-1] for c in cases]
    uniq, inv = np.unique(grp, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    single = all(len(m) == 1 for m in members)
    d = []
    while len(d) < B:
        if single:
            i = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(members), len(members))
            i = np.concatenate([members[k] for k in pick])
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return float(roc_auc_score(y, b) - roc_auc_score(y, a)), \
        [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def default_configs():
    configs = [("All four (MammoFF)", ALL4, True, "default")]
    configs += [(f"{k} alone", [v], True, "default") for k, v in ENC.items()]
    configs += [(f"Without {k}", [e for e in ALL4 if e != v], True, "default") for k, v in ENC.items()]
    configs += [("View-agnostic head only", ALL4, True, "general"),
                ("Specialist heads only", ALL4, True, "specialist"),
                ("CC view only", ALL4, True, "cc_only"),
                ("MLO view only", ALL4, True, "mlo_only"),
                ("No L2 normalisation", ALL4, False, "default")]
    return configs


def run_dataset(conn, tag, datasets, task, split_fn, configs=None):
    """configs: list of (name, encoder list, l2, branch variant); the first entry named
    'All four (MammoFF)' is the reference for every paired delta."""
    configs = configs or default_configs()
    # group by (encoder set, l2) so heads are fitted once per group and seed
    groups = {}
    for name, encs, l2, var in configs:
        groups.setdefault((tuple(encs), l2), []).append((name, var))
    per = {name: {"seeds": []} for name, *_ in configs}
    for (encs, l2), members in groups.items():
        t0 = time.time()
        ids, X, y, split, grp, view, *_ = load(conn, datasets, list(encs), task, l2)
        for seed in SEEDS:
            tr, te = split_fn(seed, split, grp, y)
            H = heads(ids, X, y, grp, view, tr, te, seed)
            for name, var in members:
                trd, _ = fuse(conn, H, ids, view, tr, var, "tr")
                ted, pc = fuse(conn, H, ids, view, te, var, "te")
                per[name]["seeds"].append(dict(tr=trd, te=ted, paired=pc))
        print(f"[{tag}] {len(members)} variant(s) on {len(encs)} extractor(s) l2={l2}: "
              f"{time.time() - t0:.0f}s", flush=True)
    ref = per["All four (MammoFF)"]["seeds"]
    res = {}
    for name, *_ in configs:
        S = per[name]["seeds"]
        view_row = name in ("CC view only", "MLO view only")
        row = {}
        for pop in (["paired"] if view_row else ["full", "paired"]):
            aucs, bals, oofs = [], [], []
            for k, s in enumerate(S):
                cases = sorted(set(s["te"]) & set(ref[k]["te"]))
                if pop == "paired":
                    cases = [c for c in cases if c in ref[k]["paired"]]
                a, b = _metrics(s["te"], cases)
                aucs.append(a); bals.append(b)
                trc = sorted(s["tr"])
                oofs.append(_metrics(s["tr"], trc)[0])
            # delta CI: CBIS seed 0 only (fixed test set); Mini pooled over seeds
            seeds_for_ci = [0] if tag == "cbis" else range(len(S))
            refd, vard, allc = {}, {}, []
            for k in seeds_for_ci:
                cases = sorted(set(S[k]["te"]) & set(ref[k]["te"]))
                if pop == "paired":
                    cases = [c for c in cases if c in ref[k]["paired"]]
                for c in cases:
                    key = f"{k}:{c}"
                    refd[key] = ref[k]["te"][c]; vard[key] = S[k]["te"][c]; allc.append(key)
            dlt, ci = delta_ci(refd, vard, allc)
            row[pop] = dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                            bal=float(np.mean(bals)), train_oof_auc=float(np.mean(oofs)),
                            delta=dlt, delta_ci=ci, n=len(cases))
        res[name] = row
        p = row.get("full", row["paired"])
        print(f"[{tag}] {name:26s} AUC {p['auc']:.3f}  delta {p['delta']:+.3f} "
              f"[{p['delta_ci'][0]:+.3f}, {p['delta_ci'][1]:+.3f}]  bal {p['bal']*100:.1f}", flush=True)
    return res


def main():
    conn = db.connect()
    out = {}
    out["cbis"] = run_dataset(conn, "cbis", ["cbis_full"], "bm",
                              lambda s, sp, g, y: (sp == "train", sp == "test"))
    json.dump(out, open(OUT, "w"), indent=2)
    out["mini"] = run_dataset(conn, "mini", ["miniddsm", "miniddsm_normal"], "nc",
                              lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s),
                                                   M.patient_test_mask(g, y, 0.2, s)))
    json.dump(out, open(OUT, "w"), indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
