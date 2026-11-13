"""Paired case-level bootstrap on the AUC difference, fine-tuned vs frozen (Table finetune),
at the headline C=0.5, identical folds and code path. 5,000 resamples, full-210 and
paired-151 populations. Writes cache/finetune_paired_bootstrap_v3.json.

Run: python finetune_paired_bootstrap.py
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import config
import finetune_eval_full4 as F4
import finetune_eval_phase2 as P2

B, SEED = 5000, 0


def _cf(fn, **kw):
    fn(**kw)
    cf = fn.last_cf
    return {str(c): (int(y), float(s)) for c, y, s in zip(cf["cids"], cf["y_true"], cf["y_score"])}, \
        {str(cf["cids"][i]) for i in cf["paired_idx"]}


def boot(base, other, cases):
    cases = sorted(cases)
    y = np.array([base[c][0] for c in cases])
    a = np.array([base[c][1] for c in cases])
    b = np.array([other[c][1] for c in cases])
    assert all(base[c][0] == other[c][0] for c in cases)
    rng = np.random.default_rng(SEED)
    d = []
    while len(d) < B:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    delta = roc_auc_score(y, b) - roc_auc_score(y, a)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(n=len(y), auc_frozen=roc_auc_score(y, a), auc_ft=roc_auc_score(y, b),
                delta=float(delta), ci=[float(lo), float(hi)])


def main():
    frozen, pf = _cf(F4.run, C=0.5, mc_fmt="mammo_clip_b5", label="Frozen 4")
    ft1, _ = _cf(F4.run, C=0.5)
    ft3, _ = _cf(P2.run, C=0.5)
    out = {}
    for name, other in (("ft1_of4", ft1), ("ft3_of4", ft3)):
        common = set(frozen) & set(other)
        out[name] = {"full": boot(frozen, other, common),
                     "paired": boot(frozen, other, common & pf)}
        for pop in ("full", "paired"):
            r = out[name][pop]
            print(f"{name} {pop:6s} n={r['n']} frozen={r['auc_frozen']:.4f} ft={r['auc_ft']:.4f} "
                  f"delta={r['delta']:+.4f} 95% CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
    path = os.path.join(config.PROJECT_DIR, "cache", "finetune_paired_bootstrap_v3.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
