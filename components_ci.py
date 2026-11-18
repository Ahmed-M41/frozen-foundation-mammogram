"""R1.8: 95% CIs for Table components (how the fourth model is used): MammoScreen score alone,
three-encoder classifier, late score fusion (w=0.6), early feature fusion (MammoFF), with a
paired bootstrap CI on each AUC difference against early fusion (5,000 resamples).
CBIS-DDSM paired subset, representative seed 0; Mini-DDSM, four seeds pooled.
Run: python components_ci.py  ->  cache/components_ci.json
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import base4_numbers as BN
import config
import db
import miniddsm_B as M
from data_scaling_extend import head_case_scores, align_B
from exp_msfeat_fusion import load_enc, BASE3, MSFEAT, cbis_ms

B = 5000


def collect(conn, datasets, split_fn, ms_map, nc, seeds):
    raw3 = load_enc(conn, datasets, BASE3); raw4 = load_enc(conn, datasets, BASE3 + [MSFEAT])
    if nc:
        ids, X3, y, split, grp, view, *_ = M.apply_task("nc", *raw3)
        keep = np.isin(raw4[7], ["NORMAL", "CANCER", "MALIGNANT"]); X4 = raw4[1][keep]
    else:
        ids, X3, y, split, grp, view = raw3[:6]; X4 = raw4[1]
    acc = {k: [] for k in ("y", "ms", "probe3", "late3", "early", "cid")}
    for seed in seeds:
        tr, te = split_fn(seed, split, grp, y)
        a_tr, a_te = head_case_scores(conn, ids, X3, y, grp, view, tr, te, M.C_HEAD, seed)
        ytr, R3tr, Mtr, yte, R3te, Mte = align_B(a_tr, a_te, ms_map, ms_map)
        b_tr, b_te = head_case_scores(conn, ids, X4, y, grp, view, tr, te, M.C_HEAD, seed)
        _, _, _, _, R4te, _ = align_B(b_tr, b_te, ms_map, ms_map)
        cid = _kept_cids(a_te, ms_map)
        assert cid == _kept_cids(b_te, ms_map) and len(cid) == len(yte)
        _, late = BN._scores("late", R3tr, Mtr, R3te, Mte)
        for k, v in (("y", yte), ("ms", Mte), ("probe3", R3te), ("late3", late), ("early", R4te),
                     ("cid", np.array(cid))):
            acc[k].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in acc.items()}


def _kept_cids(cf, msmap):
    """Case ids in the order align_B returns its test arrays."""
    cids = [str(c) for c in cf["cids"]]
    return [cids[i] for i in cf["paired_idx"] if cids[i] in msmap]


def ci_table(d):
    """Case-level bootstrap: a case pooled from several seeds' overlapping test partitions
    is resampled as one unit (plain resampling when every case occurs once)."""
    rng = np.random.default_rng(0); y = d["y"]; n = len(y)
    uniq, inv = np.unique(d["cid"], return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    single = len(uniq) == n
    boots = {k: [] for k in ("ms", "probe3", "late3", "early")}; deltas = {k: [] for k in boots}
    while len(boots["early"]) < B:
        i = rng.integers(0, n, n) if single else \
            np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[i])) < 2:
            continue
        e = roc_auc_score(y[i], d["early"][i])
        for k in boots:
            a = roc_auc_score(y[i], d[k][i]); boots[k].append(a); deltas[k].append(a - e)
    out = {}
    for k in boots:
        out[k] = dict(auc=float(roc_auc_score(y, d[k])),
                      ci=[float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))],
                      delta_vs_early=float(roc_auc_score(y, d[k]) - roc_auc_score(y, d["early"])),
                      delta_ci=[float(np.percentile(deltas[k], 2.5)), float(np.percentile(deltas[k], 97.5))])
    return out


def main():
    conn = db.connect()
    res = {}
    res["cbis_paired_seed0"] = ci_table(collect(conn, ["cbis_full"], lambda s, sp, g, y: (sp == "train", sp == "test"),
                                                cbis_ms(), False, [0]))
    res["mini_pooled"] = ci_table(collect(conn, ["miniddsm", "miniddsm_normal"],
                                          lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s), M.patient_test_mask(g, y, 0.2, s)),
                                          M.load_ms(["miniddsm", "miniddsm_normal"]), True, [0, 1, 2, 3]))
    for ds, t in res.items():
        for k, v in t.items():
            print(f"{ds:18s} {k:7s} AUC {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]  "
                  f"vs early {v['delta_vs_early']:+.3f} [{v['delta_ci'][0]:+.3f}, {v['delta_ci'][1]:+.3f}]")
    json.dump(res, open(os.path.join(config.PROJECT_DIR, "cache", "components_ci.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
