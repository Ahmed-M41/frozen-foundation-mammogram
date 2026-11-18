"""Evaluate a candidate frozen encoder against MammoFF ("why these encoders?", R2.2).

For a candidate whose embeddings are already in the database (embed_all.py), on both
datasets, under the identical headline protocol (ablation_e1.run_dataset):
  * the candidate alone
  * the candidate replacing each of the three native encoders inside MammoFF
  * the candidate added as a fifth extractor
Each row reports AUC (mean over 4 seeds), balanced accuracy at 0.5, training OOF AUC, and
a paired bootstrap 95% CI on the AUC difference against MammoFF on the same cases.

Run: python encoder_eval.py <embedder_name> [<label>]  ->  cache/encoder_eval_<name>.json
"""
import json
import os
import sys

import ablation_e1 as A
import config
import db
import miniddsm_B as M


def configs_for(cand, label):
    cfg = [("All four (MammoFF)", A.ALL4, True, "default"),
           (f"{label} alone", [cand], True, "default")]
    for k, v in A.ENC.items():
        if k == "MammoScreen":
            continue
        cfg.append((f"{label} replaces {k}", [cand if e == v else e for e in A.ALL4], True, "default"))
    cfg.append((f"MammoFF + {label}", A.ALL4 + [cand], True, "default"))
    return cfg


def main():
    cand = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else cand
    conn = db.connect()
    out = {"embedder": cand, "label": label}
    out["cbis"] = A.run_dataset(conn, "cbis", ["cbis_full"], "bm",
                                lambda s, sp, g, y: (sp == "train", sp == "test"),
                                configs_for(cand, label))
    out["mini"] = A.run_dataset(conn, "mini", ["miniddsm", "miniddsm_normal"], "nc",
                                lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s),
                                                     M.patient_test_mask(g, y, 0.2, s)),
                                configs_for(cand, label))
    path = os.path.join(config.PROJECT_DIR, "cache", f"encoder_eval_{cand}.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
