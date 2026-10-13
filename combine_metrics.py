"""Consolidate the 3 metrics (AUC, Accuracy, Balanced accuracy) into ONE image per
(ensembler, extra-dataset) — 6 images total — saved under output/combined/.

Reads the per-run CSVs:
  Ensembler B : output/data_scaling_{tag}.csv
  Ensembler A : output/data_scaling_A_{tag}.csv
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

DATASETS = [("mias", "MIAS (film)"), ("ib", "InBreast (digital)"), ("cmmd", "CMMD (digital)")]
MODELS = [("B", "Ensembler B (RAC + mammoscreen-score)", ""),
          ("A", "Ensembler A (feat + RAC + score)", "A_")]
METRICS = [("auc", "AUC", "#1f77b4"),
           ("acc", "Accuracy", "#2ca02c"),
           ("bal", "Balanced acc", "#ff7f0e")]
POINTS = list(range(0, 201, 10))
OUTDIR = os.path.join(config.OUTPUT_DIR, "combined")


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    agg = {}
    for m, _, _ in METRICS:
        mean = np.array([np.mean([r[m] for r in rows if r["pt"] == p]) for p in POINTS])
        std = np.array([np.std([r[m] for r in rows if r["pt"] == p]) for p in POINTS])
        agg[m] = (mean, std)
    return agg


def write_csv(out_csv, agg):
    """Write the exact plotted data: one row per point, mean+std of each metric."""
    fields = ["pt"]
    for m, _, _ in METRICS:
        fields += [f"{m}_mean", f"{m}_std"]
    with open(out_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        for i, p in enumerate(POINTS):
            row = {"pt": p}
            for m, _, _ in METRICS:
                mean, std = agg[m]
                row[f"{m}_mean"] = f"{mean[i]:.6f}"
                row[f"{m}_std"] = f"{std[i]:.6f}"
            wtr.writerow(row)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pts = np.array(POINTS)
    for mkey, mtitle, prefix in MODELS:
        for tag, label in DATASETS:
            path = os.path.join(config.OUTPUT_DIR, f"data_scaling_{prefix}{tag}.csv")
            if not os.path.exists(path):
                print(f"  [skip] missing {path}")
                continue
            agg = load(path)
            plt.figure(figsize=(9, 5.5))
            for m, mlabel, color in METRICS:
                mean, std = agg[m]
                plt.plot(pts, mean, "-o", ms=4, color=color, label=mlabel)
                plt.fill_between(pts, mean - std, mean + std, color=color, alpha=0.13)
            plt.axvline(100, color="gray", ls="--", lw=1)
            ymin = min(agg[m][0].min() for m, _, _ in METRICS)
            plt.text(101, ymin, f"  + {label} -->", color="gray", fontsize=9, va="bottom")
            plt.text(50, ymin, "CBIS (film)", color="gray", fontsize=9, ha="center", va="bottom")
            plt.xlabel(f"% of database used   (0-100 = CBIS;  100-200 = full CBIS + % of {label})")
            plt.ylabel("score")
            plt.title(f"{mtitle}  —  extended with {label}\n(paired-151 test, mean±std over 4 seeds)")
            plt.xticks(pts)
            plt.grid(True, alpha=0.3)
            plt.legend(loc="lower right")
            out = os.path.join(OUTDIR, f"{mkey}_{tag}.png")
            plt.tight_layout()
            plt.savefig(out, dpi=140)
            plt.close()
            out_csv = os.path.join(OUTDIR, f"{mkey}_{tag}.csv")
            write_csv(out_csv, agg)
            print("saved", out, "+", out_csv)


if __name__ == "__main__":
    main()
