"""Case-level bootstrap 95% confidence intervals for the main results (Table III and the
three-encoder row of Table sota), 5,000 resamples, percentile method.

Protocol stated in the paper:
  * CBIS-DDSM: the test set is fixed across seeds, so the CI uses one representative
    evaluation (seed 0) rather than pooling the four seeds.
  * Mini-DDSM: each seed redraws the test partition, so the four seeds' test cases are
    pooled before resampling.
Besides AUC, every operating-point metric at the fixed 0.5 threshold gets a CI too
(accuracy, balanced accuracy, sensitivity, specificity, precision, F1; reviewer R1.8).

Run: python bootstrap_ci.py   ->  cache/bootstrap_ci_v3.json
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import config
import db
import miniddsm_B as M
from data_scaling_extend import head_case_scores, align_B
from exp_msfeat_fusion import load_enc, BASE3, MSFEAT, cbis_ms

B = 5000
RNG_SEED = 0
REP_SEED = 0          # representative CBIS-DDSM evaluation
SEEDS = [0, 1, 2, 3]
THR = config.DECISION_THRESHOLD


def _point(y, s):
    p = (s >= THR).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return dict(auc=roc_auc_score(y, s), acc=(tp + tn) / len(y), bal=(sens + spec) / 2,
                sens=sens, spec=spec, prec=prec,
                f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def ci(y, s, groups=None):
    """Bootstrap CI. groups: case id per row when rows are pooled over seeds whose test
    partitions overlap; the case is then the resampling unit (all its rows together)."""
    rng = np.random.default_rng(RNG_SEED)
    est = _point(y, s)
    reps = {k: [] for k in est}
    members = None
    if groups is not None:
        g = [str(x).split(":", 1)[-1] for x in groups]
        uniq, inv = np.unique(g, return_inverse=True)
        if len(uniq) < len(g):
            members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    while len(reps["auc"]) < B:
        if members is None:
            i = rng.integers(0, len(y), len(y))
        else:
            i = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[i])) < 2:
            continue
        for k, v in _point(y[i], s[i]).items():
            reps[k].append(v)
    return {k: dict(est=float(est[k]), lo=float(np.percentile(reps[k], 2.5)),
                    hi=float(np.percentile(reps[k], 97.5))) for k in est}, len(y)


def scores(conn, datasets, split_fn, ms_map, apply_nc, enc, seeds):
    """Per seed: (y_full, s_full, y_paired, s_paired) case-level test scores."""
    raw = load_enc(conn, datasets, enc)
    if apply_nc:
        raw3 = load_enc(conn, datasets, BASE3)
        ids, _, y, split, grp, view, *_ = M.apply_task("nc", *raw3)
        keep = np.isin(raw[7], ["NORMAL", "CANCER", "MALIGNANT"])
        X = raw[1][keep]
        assert np.array_equal(ids, raw[0][keep])
    else:
        ids, X, y, split, grp, view = raw[:6]
    out = []
    for seed in seeds:
        tr, te = split_fn(seed, split, grp, y)
        cf_tr, cf_te = head_case_scores(conn, ids, X, y, grp, view, tr, te, M.C_HEAD, seed)
        _, _, _, yp, sp, _ = align_B(cf_tr, cf_te, ms_map, ms_map)
        out.append((np.array(cf_te["y_true"]), np.array(cf_te["y_score"]), yp, sp,
                    np.array([str(c) for c in cf_te["cids"]])))
    return out


def main():
    conn = db.connect()
    cbis_split = lambda s, sp, g, y: (sp == "train", sp == "test")
    mini_split = lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s),
                                      M.patient_test_mask(g, y, 0.2, s))
    res = {}
    c4 = scores(conn, ["cbis_full"], cbis_split, cbis_ms(), False, BASE3 + [MSFEAT], [REP_SEED])[0]
    c3 = scores(conn, ["cbis_full"], cbis_split, cbis_ms(), False, BASE3, [REP_SEED])[0]
    res["cbis_full_mammoff"] = ci(c4[0], c4[1])
    res["cbis_paired_mammoff"] = ci(c4[2], c4[3])
    res["cbis_paired_3enc"] = ci(c3[2], c3[3])
    mini = scores(conn, ["miniddsm", "miniddsm_normal"], mini_split,
                  M.load_ms(["miniddsm", "miniddsm_normal"]), True, BASE3 + [MSFEAT], SEEDS)
    res["mini_pooled_mammoff"] = ci(np.concatenate([m[0] for m in mini]),
                                    np.concatenate([m[1] for m in mini]),
                                    np.concatenate([m[4] for m in mini]))   # case-level
    out = {}
    for name, (d, n) in res.items():
        out[name] = dict(n=n, **d)
        print(f"{name:22s} n={n:4d}  " + "  ".join(
            f"{k} {v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for k, v in d.items()))
    path = os.path.join(config.PROJECT_DIR, "cache", "bootstrap_ci_v3.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
