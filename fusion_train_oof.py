"""Early vs late fusion on TRAINING out-of-fold case scores only (no test score is read or printed).

Checks that the choice of early feature fusion over late score fusion (Section IV-G, Table 9)
does not depend on the test sets. Same protocol as components_ci.py (Table 9): CBIS-DDSM paired
cases with seed 0 (its test set is fixed), Mini-DDSM with four seeds; late fusion is the
percentile-rank average at w=0.6. Test-side scores are computed by the shared helpers but
never used.

Run: python fusion_train_oof.py  ->  cache/fusion_train_oof.json
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


def train_only(conn, datasets, split_fn, ms_map, nc, seeds):
    raw3 = load_enc(conn, datasets, BASE3); raw4 = load_enc(conn, datasets, BASE3 + [MSFEAT])
    if nc:
        ids, X3, y, split, grp, view, *_ = M.apply_task("nc", *raw3)
        keep = np.isin(raw4[7], ["NORMAL", "CANCER", "MALIGNANT"]); X4 = raw4[1][keep]
    else:
        ids, X3, y, split, grp, view = raw3[:6]; X4 = raw4[1]
    rows = []
    for seed in seeds:
        tr, te = split_fn(seed, split, grp, y)
        a_tr, a_te = head_case_scores(conn, ids, X3, y, grp, view, tr, te, M.C_HEAD, seed)
        b_tr, b_te = head_case_scores(conn, ids, X4, y, grp, view, tr, te, M.C_HEAD, seed)
        ytr, R3tr, Mtr, _, _, _ = align_B(a_tr, a_te, ms_map, ms_map)
        ytr4, R4tr, _, _, _, _ = align_B(b_tr, b_te, ms_map, ms_map)
        assert np.array_equal(ytr, ytr4)
        late_tr, _ = BN._scores("late", R3tr, Mtr, R3tr, Mtr)   # train-side percentile fusion only
        r = dict(seed=seed, n=int(len(ytr)), pos=int(ytr.sum()),
                 ms=roc_auc_score(ytr, Mtr), probe3=roc_auc_score(ytr, R3tr),
                 late=roc_auc_score(ytr, late_tr), early=roc_auc_score(ytr, R4tr))
        rows.append(r)
        print(f"  seed {seed}: n={r['n']} pos={r['pos']}  MS {r['ms']:.3f}  3-enc {r['probe3']:.3f}  "
              f"late {r['late']:.3f}  early {r['early']:.3f}", flush=True)
    return rows


def main():
    conn = db.connect()
    out = {}
    print("CBIS-DDSM training set (paired cases), seed 0:")
    out["cbis"] = train_only(conn, ["cbis_full"], lambda s, sp, g, y: (sp == "train", sp == "test"),
                             cbis_ms(), False, [0])
    print("Mini-DDSM training partitions, seeds 0-3:")
    out["mini"] = train_only(conn, ["miniddsm", "miniddsm_normal"],
                             lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s), M.patient_test_mask(g, y, 0.2, s)),
                             M.load_ms(["miniddsm", "miniddsm_normal"]), True, [0, 1, 2, 3])
    for ds, rows in list(out.items()):
        m = {k: float(np.mean([r[k] for r in rows])) for k in ("ms", "probe3", "late", "early")}
        print(f"{ds} TRAIN-OOF mean: MS {m['ms']:.3f}  3-enc {m['probe3']:.3f}  late {m['late']:.3f}  early {m['early']:.3f}")
        out[ds + "_mean"] = m
    json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", "fusion_train_oof.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
