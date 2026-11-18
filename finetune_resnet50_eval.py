"""Evaluate the end-to-end ResNet-50 baseline (finetune_resnet50_e2e.py) against MammoFF on
the identical CBIS-DDSM official test cases: AUC, operating-point metrics at 0.5, and a
paired case-level bootstrap 95% CI on the AUC difference (5,000 resamples), for the full
210-case set and the 151 paired cases. Case score = mean of the breast's view scores.

Run: python finetune_resnet50_eval.py  ->  cache/resnet50_e2e_eval.json
"""
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

import bootstrap_ci as BC
import config
import db
from exp_msfeat_fusion import BASE3, MSFEAT, cbis_ms

CACHE = os.path.join(config.PROJECT_DIR, "cache")
TAG = os.environ.get("R50_TAG", "")


def main():
    conn = db.connect()
    pred = json.load(open(os.path.join(CACHE, f"resnet50_e2e{TAG}_preds.json")))
    log = json.load(open(os.path.join(CACHE, f"resnet50_e2e{TAG}_log.json")))
    case = {r["id"]: (r["case_id"], r["binary_label"], r["view"]) for r in conn.execute(
        "SELECT id, case_id, binary_label, view FROM samples WHERE dataset='cbis_full' AND split='test'")}
    by_case = defaultdict(list); lab = {}; views = defaultdict(set)
    for sid, s in pred["test"].items():
        c, y, v = case[int(sid)]
        by_case[c].append(s); lab[c] = max(lab.get(c, 0), y); views[c].add(v)
    r50 = {c: float(np.mean(v)) for c, v in by_case.items()}
    # MammoFF case scores (representative seed 0), keyed by case id
    split = lambda s, sp, g, y: (sp == "train", sp == "test")
    raw = BC.scores(conn, ["cbis_full"], split, cbis_ms(), False, BASE3 + [MSFEAT], [0])[0]
    from data_scaling_extend import head_case_scores  # noqa: F401  (documented dependency)
    import ablation_e1 as A
    ids, X, y, sp, grp, view, *_ = A.load(conn, ["cbis_full"], A.ALL4, "bm")
    tr, te = sp == "train", sp == "test"
    H = A.heads(ids, X, y, grp, view, tr, te, 0)
    mff, paired = A.fuse(conn, H, ids, view, te, "default", "te")
    out = {}
    for pop in ("full", "paired"):
        cases = sorted(c for c in r50 if c in mff and (pop == "full" or c in paired))
        yy = np.array([lab[c] for c in cases]); a = np.array([mff[c][1] for c in cases])
        b = np.array([r50[c] for c in cases])
        ci_r50, n = BC.ci(yy, b)
        rng = np.random.default_rng(0); d = []
        while len(d) < 5000:
            i = rng.integers(0, len(yy), len(yy))
            if len(np.unique(yy[i])) == 2:
                d.append(roc_auc_score(yy[i], b[i]) - roc_auc_score(yy[i], a[i]))
        out[pop] = dict(n=n, resnet50=ci_r50, mammoff_auc=float(roc_auc_score(yy, a)),
                        delta=float(roc_auc_score(yy, b) - roc_auc_score(yy, a)),
                        delta_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        r = out[pop]
        print(f"{pop:6s} n={n} ResNet-50 AUC {r['resnet50']['auc']['est']:.3f} "
              f"[{r['resnet50']['auc']['lo']:.3f}, {r['resnet50']['auc']['hi']:.3f}]  "
              f"MammoFF {r['mammoff_auc']:.3f}  diff {r['delta']:+.3f} "
              f"[{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}]  "
              f"acc {r['resnet50']['acc']['est']*100:.1f} bal {r['resnet50']['bal']['est']*100:.1f}")
    out["gpu_hours"] = sum(e["wall_clock_sec"] for e in log) / 3600
    out["peak_gpu_gb"] = max(e["peak_gpu_gb"] for e in log)
    out["config"] = pred["config"]
    print(f"GPU-hours {out['gpu_hours']:.2f}, peak {out['peak_gpu_gb']:.1f} GB")
    json.dump(out, open(os.path.join(CACHE, f"resnet50_e2e{TAG}_eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
