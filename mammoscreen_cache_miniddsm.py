"""Cache mammoscreen zero-shot case scores for Mini-DDSM (lesion + normal),
train+test, so the fusion threshold can be tuned honestly on train.

mammoscreen sees the same 512x512 breast-cropped jpegs the backbones do
(crop=False here — the crop was baked in at ingest by prep_miniddsm). One npz per
(dataset, split): cache/mammoscreen_<dataset>_<split>.npz with cids,y,s,pr.

Run (after prep_miniddsm.py):  python mammoscreen_cache_miniddsm.py
"""
import os

import numpy as np

import config
import mammoscreen

OUT = os.path.join(config.PROJECT_DIR, "cache")
DATASETS = ["miniddsm", "miniddsm_normal"]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for dataset in DATASETS:
        for split in ["test", "train"]:
            cids, y, s, pr = mammoscreen.case_scores(dataset=dataset, split=split,
                                                     paired_only=True, crop=False)
            np.savez(os.path.join(OUT, f"mammoscreen_{dataset}_{split}.npz"),
                     cids=cids, y=y, s=s, pr=pr)
            print(f"  {dataset}/{split}: cached {len(cids)} paired cases "
                  f"(pos={int(np.sum(y))})")
    print("MAMMOSCREEN_MINIDDSM_CACHE_DONE")
