"""Cluster (case-level) bootstrap for results pooled over the four seeds (reviewer R1.8).

Mini-DDSM and INbreast redraw a 20% patient-stratified test partition per seed, so the
four partitions overlap and one breast can be scored up to four times. Resampling the
pooled rows as if independent would make the intervals too narrow. Here the resampling
unit is the case (one breast): a case is drawn with replacement and ALL of its pooled
evaluations enter the replicate, which matches the paper's case-level bootstrap.

ci(y, s, groups)                 -> dict metric -> {est, lo, hi}   (same metrics as bootstrap_ci)
diff_ci(y, a, b, groups)          -> paired AUC difference b - a with its 95% CI
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import bootstrap_ci as BC

B = 5000
RNG_SEED = 0


def _index(groups):
    groups = np.asarray([str(g) for g in groups])
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    return members


def _draw(rng, members):
    pick = rng.integers(0, len(members), len(members))
    return np.concatenate([members[k] for k in pick])


def ci(y, s, groups):
    y, s = np.asarray(y), np.asarray(s)
    members = _index(groups)
    rng = np.random.default_rng(RNG_SEED)
    est = BC._point(y, s)
    reps = {k: [] for k in est}
    while len(reps["auc"]) < B:
        i = _draw(rng, members)
        if len(np.unique(y[i])) < 2:
            continue
        for k, v in BC._point(y[i], s[i]).items():
            reps[k].append(v)
    return {k: dict(est=float(est[k]), lo=float(np.percentile(reps[k], 2.5)),
                    hi=float(np.percentile(reps[k], 97.5))) for k in est}


def diff_ci(y, a, b, groups):
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    members = _index(groups)
    rng = np.random.default_rng(RNG_SEED)
    d = []
    while len(d) < B:
        i = _draw(rng, members)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(roc_auc_score(y, b) - roc_auc_score(y, a)), lo=float(lo), hi=float(hi))
