"""Confusion matrices for Fig. cm: one representative seed (seed 0, the same seed the
submitted figure used), at the fixed 0.5 threshold, from base4_numbers_v3.json.

Run after base4_numbers.py:  python make_confusion_json.py -> cache/confusion_singleseed_v3.json
"""
import json
import os

import config

SEED = 0
src = json.load(open(os.path.join(config.PROJECT_DIR, "cache", "base4_numbers_v3.json")))
out = {}
for key, label in (("cbis_bm", "paired, single representative seed"),
                   ("miniddsm_nc", "single representative seed")):
    r = src[key]["paired"]["base4"]["per_seed"][SEED]
    out[key] = dict(tn=r["tn"], fp=r["fp"], fn=r["fn"], tp=r["tp"], auc=r["auc"],
                    bal_acc=r["bal_acc"], n=r["tn"] + r["fp"] + r["fn"] + r["tp"],
                    label=label + f" (seed {SEED}, threshold {config.DECISION_THRESHOLD})")
path = os.path.join(config.PROJECT_DIR, "cache", "confusion_singleseed_v3.json")
json.dump(out, open(path, "w"), indent=2)
print("saved", path, {k: (v["tn"], v["fp"], v["fn"], v["tp"]) for k, v in out.items()})
