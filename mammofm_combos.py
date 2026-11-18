"""Follow-up on Mammo-FM (batmanLab/Mammo-FM), the strongest public mammography encoder in the
Nguyen et al. 2026 benchmark, never trained on CBIS-DDSM/DDSM.

Candidate encoder sets are compared under the paper's exact protocol (same heads, C=0.5,
threshold 0.5, CBIS-DDSM official split seed 0..3, Mini-DDSM with shared-film patients removed,
4 seeds). Selection rule, fixed BEFORE looking at test numbers: the set with the highest
TRAINING out-of-fold AUC (train_oof_auc) is the one that would be chosen; test AUCs with paired
CIs against MammoFF are reported for every set so nothing is hidden.

Run: python mammofm_combos.py  ->  cache/mammofm_combos.json
"""
import json
import os

import ablation_e1 as A
import config
import db
import miniddsm_B as M

MCLIP, DINO, RAD, MS = A.ALL4          # mammo_clip_b5, dinov2, rad_dino, mammoscreen_feat
FM = "mammo_fm"
OUT = os.path.join(config.PROJECT_DIR, "cache", "mammofm_combos.json")


def configs():
    sets = [("All four (MammoFF)", A.ALL4),
            ("Mammo-FM alone", [FM]),
            ("Mammo-FM replaces Mammo-CLIP", [FM, DINO, RAD, MS]),
            ("Mammo-FM replaces RAD-DINO", [MCLIP, DINO, FM, MS]),
            ("Mammo-FM replaces both", [FM, DINO, MS]),
            ("MammoFF + Mammo-FM (five)", A.ALL4 + [FM]),
            ("Mammo-FM + MammoScreen", [FM, MS]),
            ("Mammo-FM + DINOv2", [FM, DINO]),
            ("Three without MammoScreen, Mammo-FM for Mammo-CLIP", [FM, DINO, RAD])]
    return [(n, e, True, "default") for n, e in sets]


def main():
    assert A.ALL4 == [MCLIP, DINO, RAD, MS] and "mammo_clip" in MCLIP and "mammoscreen" in MS
    conn = db.connect()
    out = {}
    out["cbis"] = A.run_dataset(conn, "cbis", ["cbis_full"], "bm",
                                lambda s, sp, g, y: (sp == "train", sp == "test"), configs())
    json.dump(out, open(OUT, "w"), indent=2)
    out["mini_dedup"] = A.run_dataset(conn, "mini", ["miniddsm", "miniddsm_normal"], "nc",
                                      lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s),
                                                           M.patient_test_mask(g, y, 0.2, s)),
                                      configs())
    out["mini_excluded_shared_films"] = bool(config.MINI_EXCLUDE_SHARED_FILMS)
    json.dump(out, open(OUT, "w"), indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
