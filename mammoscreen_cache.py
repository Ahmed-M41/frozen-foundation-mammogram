"""Cache mammoscreen zero-shot case scores for CBIS train+test paired cases,
so the fusion threshold can be tuned honestly on train (not on the test set)."""
import os

import numpy as np

import config
import mammoscreen

OUT = os.path.join(config.PROJECT_DIR, "cache")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for split in ["test", "train"]:
        cids, y, s, pr = mammoscreen.case_scores(dataset="cbis_full", split=split, paired_only=True)
        np.savez(os.path.join(OUT, f"mammoscreen_{split}.npz"), cids=cids, y=y, s=s, pr=pr)
        print(f"  {split}: cached {len(cids)} paired cases")
    print("MAMMOSCREEN_CACHE_DONE")
