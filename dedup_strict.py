"""Stricter de-duplication check (R2.4): Mini-DDSM with patients removed at >=10 keypoint matches
(also drops the mixed 10-14 band), same re-split protocol as the main result (removal before the
patient-level split). Run: python dedup_strict.py -> cache/dedup_strict.json"""
import json
import os

import config
config.MINI_SHARED_MIN_INLIERS = 10
import ablation_e1 as A
import db
import miniddsm_B as M

cfg = [("All four (MammoFF)", A.ALL4, True, "default"),
       ("Without MammoScreen", [e for e in A.ALL4 if e != "mammoscreen_feat"], True, "default")]
conn = db.connect()
print("patients removed:", len(M.shared_film_patients(conn)))
res = A.run_dataset(conn, "mini-strict10", ["miniddsm", "miniddsm_normal"], "nc",
                    lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s), M.patient_test_mask(g, y, 0.2, s)), cfg)
json.dump(dict(min_inliers=10, n_patients_removed=len(M.shared_film_patients(conn)), results=res),
          open(os.path.join(config.PROJECT_DIR, "cache", "dedup_strict.json"), "w"), indent=2)
