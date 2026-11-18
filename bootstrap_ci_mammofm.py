"""Clean variant (no CBIS-DDSM pre-training anywhere): Mammo-CLIP + DINOv2 + RAD-DINO + Mammo-FM,
CBIS-DDSM paired subset, same protocol as the three-encoder row of Table sota:
  * point estimates (AUC, accuracy at 0.5) = mean over the four seeds;
  * 95% CI = case-level bootstrap (5,000 resamples) on the representative seed 0.
Run: python bootstrap_ci_mammofm.py -> cache/bootstrap_ci_mammofm.json"""
import json
import os

import numpy as np

import config
import db
from bootstrap_ci import ci, scores, _point, REP_SEED, SEEDS
from exp_msfeat_fusion import BASE3, cbis_ms


def main():
    conn = db.connect()
    cbis_split = lambda s, sp, g, y: (sp == "train", sp == "test")
    out = {}
    for name, enc in (("cbis_paired_3enc", BASE3), ("cbis_paired_3enc_mammofm", BASE3 + ["mammo_fm"])):
        runs = scores(conn, ["cbis_full"], cbis_split, cbis_ms(), False, enc, SEEDS)
        assert REP_SEED == SEEDS[0]
        d, n = ci(runs[0][2], runs[0][3])
        pts = [_point(r[2], r[3]) for r in runs]
        mean = {k: float(np.mean([p[k] for p in pts])) for k in pts[0]}
        out[name] = dict(n=n, seed0_ci=d, mean4=mean)
        print(f"{name:26s} n={n}  mean4 auc {mean['auc']:.3f} acc {mean['acc']:.3f} bal {mean['bal']:.3f} | "
              f"seed0 auc {d['auc']['est']:.3f} [{d['auc']['lo']:.3f}, {d['auc']['hi']:.3f}]", flush=True)
    path = os.path.join(config.PROJECT_DIR, "cache", "bootstrap_ci_mammofm.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
