"""Full metric set with 95% CIs (seeds pooled, threshold 0.5) for MammoFF and the three
encoders on Mini-DDSM normal-vs-cancer after removing every patient with a film shared with
CBIS-DDSM (>=15 keypoint matches; dedup_eval.py). -> cache/dedup_metrics.json"""
import json
import os

import numpy as np

import ablation_e1 as A
import bootstrap_ci as BC
import config
import db
import miniddsm_B as M
from dedup_eval import excluded_patients

conn = db.connect()
ex = excluded_patients(conn, 15)


def split(s, sp, g, y):
    keep = ~np.isin(g, list(ex)); te = M.patient_test_mask(g, y, 0.2, s)
    return (~te) & keep, te & keep


out = {}
for name, enc in (("MammoFF", A.ALL4), ("3 encoders", [e for e in A.ALL4 if e != "mammoscreen_feat"])):
    sc = BC.scores(conn, ["miniddsm", "miniddsm_normal"], split, M.load_ms(["miniddsm", "miniddsm_normal"]),
                   True, enc, A.SEEDS)
    y = np.concatenate([s[0] for s in sc]); s_ = np.concatenate([s[1] for s in sc])
    ci, n = BC.ci(y, s_)
    out[name] = dict(n_pooled=n, **ci)
    print(name, n, {k: f"{v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for k, v in ci.items()}, flush=True)
json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", "dedup_metrics.json"), "w"), indent=2)
