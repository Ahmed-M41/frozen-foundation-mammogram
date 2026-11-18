"""E10 (reviewers R1.1, R1.2): dataset composition per partition and a patient-overlap check.

CBIS-DDSM (mass subset, benign vs malignant): official train/test split.
Mini-DDSM (normal vs cancer; benign breasts excluded): patient-stratified 20% test partition,
redrawn per seed; counts shown for each of the four seeds.
For every partition: patients, breasts (cases), images, breasts with both views, and class
counts at breast level. Also asserts that no patient appears in both partitions.

Run: python dataset_composition.py  ->  cache/dataset_composition.json
"""
import json
import os

import numpy as np

import config
import db
import miniddsm_B as M
from exp_msfeat_fusion import load_enc, BASE3

SEEDS = [0, 1, 2, 3]


def describe(ids, y, grp, view, conn, mask):
    rows = conn.execute(f"SELECT id, case_id FROM samples WHERE id IN ({','.join(str(int(i)) for i in ids[mask])})").fetchall()
    case_of = {r["id"]: r["case_id"] for r in rows}
    cases = {}
    for i, lab, v in zip(ids[mask], y[mask], view[mask]):
        c = case_of[int(i)]
        d = cases.setdefault(c, {"y": 0, "views": set()})
        d["y"] = max(d["y"], int(lab)); d["views"].add(v)
    return dict(patients=int(len(np.unique(grp[mask]))), breasts=len(cases), images=int(mask.sum()),
                breasts_both_views=sum(len(d["views"]) == 2 for d in cases.values()),
                breasts_positive=sum(d["y"] for d in cases.values()),
                breasts_negative=sum(1 - d["y"] for d in cases.values()))


def main():
    conn = db.connect()
    out = {}
    ids, X, y, split, grp, view, *_ = load_enc(conn, ["cbis_full"], BASE3)
    tr, te = split == "train", split == "test"
    shared = set(grp[tr]) & set(grp[te])
    out["cbis"] = dict(train=describe(ids, y, grp, view, conn, tr), test=describe(ids, y, grp, view, conn, te),
                       shared_patients=len(shared))
    raw = load_enc(conn, ["miniddsm", "miniddsm_normal"], BASE3)
    ids, X, y, split, grp, view, *_ = M.apply_task("nc", *raw)
    out["mini"] = {"total": describe(ids, y, grp, view, conn, np.ones(len(ids), bool))}
    for s in SEEDS:
        te = M.patient_test_mask(grp, y, 0.2, s)
        out["mini"][f"seed{s}"] = dict(train=describe(ids, y, grp, view, conn, ~te),
                                       test=describe(ids, y, grp, view, conn, te),
                                       shared_patients=len(set(grp[te]) & set(grp[~te])))
    path = os.path.join(config.PROJECT_DIR, "cache", "dataset_composition.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
