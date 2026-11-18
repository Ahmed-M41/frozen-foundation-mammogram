"""E8 (reviewers R1.9, R2.4): MammoFF on CMMD, a full-field DIGITAL mammography dataset
(Chinese Mammography Database, GE Senographe; D1 biopsy cohort: 1,026 patients, 2,214
images, benign vs malignant), trained and tested entirely within CMMD.

Uses features already stored in the database (the source PNGs are no longer on disk):
Mammo-CLIP, RAD-DINO, MammoScreen and DINOv2 ViT-L/14. DINOv2 is available for CMMD only at
its default 224x224 input (the headline uses 518x518), so this is MammoFF with a
lower-resolution DINOv2; reported as such. None of the four extractors was pre-trained on
CMMD (MammoScreen: CBIS-DDSM + RSNA; Mammo-CLIP: UPMC + VinDr).

Protocol identical to Mini-DDSM: patient-stratified 20% test partition redrawn for each of
four seeds, patient-grouped 5-fold heads (C=0.5), fixed 0.5 threshold, paired bootstrap CIs
on the AUC difference (seeds pooled). View labels come from filename order (CC, MLO per
breast; documented heuristic in prep_cmmd.py).

Run: python cmmd_eval.py  ->  cache/cmmd_eval.json
"""
import json
import os

import numpy as np

import ablation_e1 as A
import bootstrap_ci as BC
import config
import db
import miniddsm_B as M

ENC = {"Mammo-CLIP": "mammo_clip_b5", "DINOv2 (224)": "dinov2_vitl14_noclahe",
       "RAD-DINO": "rad_dino_noclahe", "MammoScreen": "mammoscreen_feat"}
ALL4 = list(ENC.values())


def main():
    conn = db.connect()
    cfg = [("All four (MammoFF)", ALL4, True, "default")]
    cfg += [(f"{k} alone", [v], True, "default") for k, v in ENC.items()]
    cfg += [(f"Without {k}", [e for e in ALL4 if e != v], True, "default") for k, v in ENC.items()]
    cfg += [("View-agnostic head only", ALL4, True, "general")]
    split = lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s), M.patient_test_mask(g, y, 0.2, s))
    res = A.run_dataset(conn, "cmmd", ["cmmd_full"], "bm", split, cfg)
    # full metric set with CIs for the main configuration (seeds pooled, threshold 0.5)
    sc = BC.scores(conn, ["cmmd_full"], split, {}, False, ALL4, A.SEEDS)
    y = np.concatenate([s[0] for s in sc]); s_ = np.concatenate([s[1] for s in sc])
    ci, n = BC.ci(y, s_)
    out = {"ablation": res, "main_metrics_ci": dict(n=n, **ci),
           "n_patients": int(conn.execute("SELECT COUNT(DISTINCT patient_id) FROM samples WHERE dataset='cmmd_full'").fetchone()[0]),
           "n_images": int(conn.execute("SELECT COUNT(*) FROM samples WHERE dataset='cmmd_full'").fetchone()[0]),
           "malignant_images": int(conn.execute("SELECT SUM(binary_label) FROM samples WHERE dataset='cmmd_full'").fetchone()[0])}
    path = os.path.join(config.PROJECT_DIR, "cache", "cmmd_eval.json")
    json.dump(out, open(path, "w"), indent=2)
    print("main:", {k: f"{v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for k, v in ci.items()})
    print("saved", path)


if __name__ == "__main__":
    main()
