"""Overlay Ensembler A vs Ensembler B under database extension, per extra dataset.
Reads output/data_scaling_A_{tag}.csv (A) and output/data_scaling_{tag}.csv (B).

Two figures per dataset:
  absolute AUC vs #extra images (shows A sits higher; both trends)
  delta-AUC vs full-CBIS baseline (shows who is more affected by the extra data)
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

SERIES = [("mias", "MIAS (film)"), ("ib", "InBreast (digital)"), ("cmmd", "CMMD (digital)")]


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    pts = sorted({r["pt"] for r in rows})
    xs, ys, es = [], [], []
    for p in pts:
        if p < 100:
            continue
        sub = [r for r in rows if r["pt"] == p]
        xs.append(np.mean([r["n_ib"] for r in sub]))
        ys.append(np.mean([r["auc"] for r in sub]))
        es.append(np.std([r["auc"] for r in sub]))
    return np.array(xs), np.array(ys), np.array(es)


def main():
    print("=== Ensembler A vs B: AUC delta(full-100%) per extra dataset ===")
    for tag, label in SERIES:
        pA = os.path.join(config.OUTPUT_DIR, f"data_scaling_A_{tag}.csv")
        pB = os.path.join(config.OUTPUT_DIR, f"data_scaling_{tag}.csv")
        if not (os.path.exists(pA) and os.path.exists(pB)):
            print(f"  [skip {tag}] missing csv")
            continue
        xA, yA, eA = load(pA)
        xB, yB, eB = load(pB)

        # absolute AUC
        plt.figure(figsize=(8.5, 5))
        plt.plot(xA, yA, "-o", color="#1f77b4", label="Ensembler A (feat+RAC+score)")
        plt.fill_between(xA, yA - eA, yA + eA, color="#1f77b4", alpha=0.12)
        plt.plot(xB, yB, "-o", color="#d62728", label="Ensembler B (RAC+score)")
        plt.fill_between(xB, yB - eB, yB + eB, color="#d62728", alpha=0.12)
        plt.xlabel(f"number of EXTRA {label} images added to CBIS")
        plt.ylabel("Case-level AUC")
        plt.title(f"A vs B under {label} augmentation (paired-151, 4 seeds)")
        plt.grid(True, alpha=0.3); plt.legend()
        out = os.path.join(config.OUTPUT_DIR, f"AvsB_{tag}_auc.png")
        plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
        print("  saved", out)

        # delta
        plt.figure(figsize=(8.5, 5))
        plt.plot(xA, yA - yA[0], "-o", color="#1f77b4", label="Ensembler A")
        plt.fill_between(xA, yA - yA[0] - eA, yA - yA[0] + eA, color="#1f77b4", alpha=0.12)
        plt.plot(xB, yB - yB[0], "-o", color="#d62728", label="Ensembler B")
        plt.fill_between(xB, yB - yB[0] - eB, yB - yB[0] + eB, color="#d62728", alpha=0.12)
        plt.axhline(0, color="gray", ls="--", lw=1)
        plt.xlabel(f"number of EXTRA {label} images added to CBIS")
        plt.ylabel("Δ AUC vs. full-CBIS baseline")
        plt.title(f"Who is more affected by {label}? A vs B (paired-151, 4 seeds)")
        plt.grid(True, alpha=0.3); plt.legend()
        out = os.path.join(config.OUTPUT_DIR, f"AvsB_{tag}_delta.png")
        plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
        print("  saved", out)

        print(f"  {label:20s}: A base={yA[0]:.4f} full={yA[-1]:.4f} (d{yA[-1]-yA[0]:+.4f}) | "
              f"B base={yB[0]:.4f} full={yB[-1]:.4f} (d{yB[-1]-yB[0]:+.4f})")


if __name__ == "__main__":
    main()
