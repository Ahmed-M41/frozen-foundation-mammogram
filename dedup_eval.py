"""E3 step 4 (reviewer R2.4): Mini-DDSM normal-vs-cancer re-run with every patient who has
a film shared with CBIS-DDSM removed (duplicates from duplicate_orb.py).

Two exclusion levels: confirmed shared films (>=15 geometrically consistent keypoint
matches, visually verified) and a conservative level that also drops the uncertain band
(>=10). Identical protocol to the headline (4 seeds, patient-stratified 20% test, C=0.5,
threshold 0.5); paired CIs are against the full method on the same de-duplicated cases.

Run: python dedup_eval.py  ->  cache/dedup_eval.json
"""
import json
import os

import numpy as np

import ablation_e1 as A
import config
import db
import miniddsm_B as M


def excluded_patients(conn, thr):
    pairs = json.load(open(os.path.join(config.PROJECT_DIR, "cache", "duplicate_orb.json")))["pairs"]
    files = {r["mini"] for r in pairs if r["inliers"] >= thr}
    rows = conn.execute("SELECT file_name, patient_id FROM samples WHERE dataset IN ('miniddsm','miniddsm_normal')")
    return {r["patient_id"] for r in rows if r["file_name"] in files}


def main():
    conn = db.connect()
    cfg = [("All four (MammoFF)", A.ALL4, True, "default"),
           ("Without MammoScreen", [e for e in A.ALL4 if e != "mammoscreen_feat"], True, "default"),
           ("MammoScreen alone", ["mammoscreen_feat"], True, "default"),
           ("Mammo-CLIP alone", ["mammo_clip_b5"], True, "default")]
    out = {}
    for thr in (15, 10):
        ex = excluded_patients(conn, thr)

        def split(s, sp, g, y, ex=ex):
            keep = ~np.isin(g, list(ex))
            te = M.patient_test_mask(g, y, 0.2, s)
            return (~te) & keep, te & keep
        print(f"--- exclusion >= {thr} inliers: {len(ex)} Mini-DDSM patients removed", flush=True)
        res = A.run_dataset(conn, f"mini-dedup{thr}", ["miniddsm", "miniddsm_normal"], "nc", split, cfg)
        out[f"thr{thr}"] = dict(n_patients_removed=len(ex), results=res)
    json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", "dedup_eval.json"), "w"), indent=2)
    print("saved")


if __name__ == "__main__":
    main()
