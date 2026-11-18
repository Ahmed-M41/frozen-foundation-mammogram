"""E7 (reviewer R1.10): operating points that prioritise sensitivity.

For target sensitivities of 90% and 95%, the threshold is set on the TRAINING out-of-fold
case scores (the highest threshold whose training sensitivity reaches the target) and
applied unchanged to the test set. Reports test sensitivity and specificity, mean over the
four seeds, for MammoFF on CBIS-DDSM (full and paired) and Mini-DDSM.

Uses the per-seed case scores cached by threshold_rules.py.
Run: python high_sensitivity_points.py  ->  cache/high_sensitivity_points.json
"""
import json
import os
import pickle

import numpy as np

import config

TARGETS = [0.90, 0.95]


def thr_for_sens(y, s, target):
    pos = np.sort(s[y == 1])[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[k])          # predict malignant if score >= thr -> training sens >= target


def main():
    cbis, mini = pickle.load(open(os.path.join(config.PROJECT_DIR, "cache",
                                               "threshold_rules_scores.pkl"), "rb"))
    out = {}
    for name, per_seed, pop in (("cbis_full", cbis, "full"), ("cbis_paired", cbis, "paired"),
                                ("mini", mini, "full")):
        out[name] = {}
        for t in TARGETS:
            rows = []
            for d in per_seed:
                ytr, str_, yte, ste = d[pop][:4]
                thr = thr_for_sens(ytr, str_, t)
                p = ste >= thr
                rows.append(dict(thr=thr, sens=float(p[yte == 1].mean()),
                                 spec=float((~p[yte == 0]).mean()),
                                 train_sens=float((str_[ytr == 1] >= thr).mean())))
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            out[name][f"{int(t*100)}"] = agg
            print(f"{name:12s} target sens {t:.0%}: threshold {agg['thr']:.3f}  "
                  f"test sens {agg['sens']*100:.1f}%  test spec {agg['spec']*100:.1f}%")
    path = os.path.join(config.PROJECT_DIR, "cache", "high_sensitivity_points.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
