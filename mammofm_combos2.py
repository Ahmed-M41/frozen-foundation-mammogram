"""Follow-up to mammofm_combos.py (Ahmed's question): Mammo-FM in place of MammoScreen, i.e. four
encoders with no CBIS-DDSM pre-training at all. Same protocol and training-OOF selection rule.
Run: python mammofm_combos2.py -> cache/mammofm_combos2.json"""
import json
import os

import ablation_e1 as A
import config
import db
import miniddsm_B as M

MCLIP, DINO, RAD, MS = A.ALL4
FM = "mammo_fm"
OUT = os.path.join(config.PROJECT_DIR, "cache", "mammofm_combos2.json")
SETS = [("All four (MammoFF)", A.ALL4),
        ("Mammo-FM replaces MammoScreen", [MCLIP, DINO, RAD, FM]),
        ("Mammo-FM replaces MammoScreen, no RAD-DINO", [MCLIP, DINO, FM]),
        ("Three encoders (no MammoScreen)", [MCLIP, DINO, RAD])]


def main():
    assert "mammoscreen" in MS
    cfg = [(n, e, True, "default") for n, e in SETS]
    conn = db.connect()
    out = {"cbis": A.run_dataset(conn, "cbis", ["cbis_full"], "bm",
                                 lambda s, sp, g, y: (sp == "train", sp == "test"), cfg),
           "mini_dedup": A.run_dataset(conn, "mini", ["miniddsm", "miniddsm_normal"], "nc",
                                       lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s),
                                                            M.patient_test_mask(g, y, 0.2, s)), cfg)}
    json.dump(out, open(OUT, "w"), indent=2)
    for ds, r in out.items():
        for k, v in r.items():
            f, p = v["full"], v["paired"]
            print(f"{ds:10s} {k:45s} trainOOF {f['train_oof_auc']:.4f} test {f['auc']:.3f} "
                  f"diff {f['delta']:+.3f} [{f['delta_ci'][0]:+.3f},{f['delta_ci'][1]:+.3f}] bal {f['bal']*100:.1f}"
                  + (f" | paired {p['auc']:.3f} {p['delta']:+.3f} [{p['delta_ci'][0]:+.3f},{p['delta_ci'][1]:+.3f}]" if ds == "cbis" else ""))


if __name__ == "__main__":
    main()
