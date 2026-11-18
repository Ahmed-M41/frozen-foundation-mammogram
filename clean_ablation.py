"""Leave-one-out inside the CLEAN configurations (no CBIS-DDSM-pre-trained extractor):
three native encoders, and the Mammo-FM variant. Paired CIs are against the clean reference.
Selection by training out-of-fold AUC. Run: python clean_ablation.py -> cache/clean_ablation.json"""
import json
import os

import ablation_e1 as A
import config
import db
import miniddsm_B as M

MCLIP, DINO, RAD, MS = A.ALL4
FM = "mammo_fm"
SETS3 = [("Clean-3: Mammo-CLIP + DINOv2 + RAD-DINO", [MCLIP, DINO, RAD]),
         ("Clean-3 without Mammo-CLIP", [DINO, RAD]),
         ("Clean-3 without DINOv2", [MCLIP, RAD]),
         ("Clean-3 without RAD-DINO", [MCLIP, DINO])]
SETSFM = [("Clean-FM: Mammo-CLIP + DINOv2 + RAD-DINO + Mammo-FM", [MCLIP, DINO, RAD, FM]),
          ("Clean-FM without Mammo-CLIP", [DINO, RAD, FM]),
          ("Clean-FM without DINOv2", [MCLIP, RAD, FM]),
          ("Clean-FM without RAD-DINO", [MCLIP, DINO, FM]),
          ("Clean-FM without Mammo-FM", [MCLIP, DINO, RAD])]
SPL = {"cbis": (["cbis_full"], "bm", lambda s, sp, g, y: (sp == "train", sp == "test")),
       "mini": (["miniddsm", "miniddsm_normal"], "nc",
                lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s), M.patient_test_mask(g, y, 0.2, s)))}


def main():
    conn = db.connect(); out = {}
    for name, sets in (("clean3", SETS3), ("cleanfm", SETSFM)):
        # run_dataset uses the row named "All four (MammoFF)" as the paired reference; here that
        # slot holds the CLEAN reference set, relabelled after the run.
        ref = sets[0][0]
        cfg = [("All four (MammoFF)" if i == 0 else n, e, True, "default") for i, (n, e) in enumerate(sets)]
        for ds, (dsets, task, split) in SPL.items():
            r = A.run_dataset(conn, f"{name}-{ds}", dsets, task, split, cfg)
            out[f"{name}_{ds}"] = {(ref if k == "All four (MammoFF)" else k): v for k, v in r.items()}
    json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", "clean_ablation.json"), "w"), indent=2)
    for key, r in out.items():
        for k, v in r.items():
            f, p = v["full"], v["paired"]
            print(f"{key:13s} {k:52s} trainOOF {f['train_oof_auc']:.3f} test {f['auc']:.3f} "
                  f"{f['delta']:+.3f} [{f['delta_ci'][0]:+.3f},{f['delta_ci'][1]:+.3f}]"
                  + (f" | paired {p['auc']:.3f} {p['delta']:+.3f} [{p['delta_ci'][0]:+.3f},{p['delta_ci'][1]:+.3f}]" if key.endswith("cbis") else ""))


if __name__ == "__main__":
    main()
