"""Film (MIAS) vs digital (InBreast) head-to-head: does the DOMAIN of the added
data matter? Both were added, in isolation, on top of the SAME full-CBIS baseline
(Ensembler B). We plot the change relative to the 100%-CBIS baseline against the
NUMBER of extra images added, so the two are compared at MATCHED dose (MIAS only
reaches ~115 imgs; InBreast reaches ~409).

Reads output/data_scaling_{mias,ib}.csv -> output/compare_film_vs_digital_{auc,bal}.png
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def load(tag):
    rows = []
    with open(os.path.join(config.OUTPUT_DIR, f"data_scaling_{tag}.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    pts = sorted({r["pt"] for r in rows})
    # keep the extension region (pt>=100): x = mean #extra imgs, y = mean metric
    out = {}
    for metric in ["auc", "bal"]:
        xs, ys, es = [], [], []
        for p in pts:
            if p < 100:
                continue
            sub = [r for r in rows if r["pt"] == p]
            xs.append(np.mean([r["n_ib"] for r in sub]))
            ys.append(np.mean([r[metric] for r in sub]))
            es.append(np.std([r[metric] for r in sub]))
        out[metric] = (np.array(xs), np.array(ys), np.array(es))
    return out


SERIES = [("mias", "MIAS (film)", "#2ca02c"),
          ("ib", "InBreast (digital)", "#1f77b4"),
          ("cmmd", "CMMD (digital)", "#d62728")]


def main():
    data = {tag: load(tag) for tag, _, _ in SERIES}
    names = {"auc": "Case-level AUC", "bal": "Balanced accuracy"}
    for metric in ["auc", "bal"]:
        plt.figure(figsize=(9, 5.5))
        for tag, lab, color in SERIES:
            x, y, e = data[tag][metric]
            base = y[0]                                   # value at 0 extra (= full CBIS)
            plt.plot(x, y - base, "-o", color=color, label=lab)
            plt.fill_between(x, y - base - e, y - base + e, color=color, alpha=0.12)
        plt.axhline(0, color="gray", ls="--", lw=1)
        plt.xlabel("number of EXTRA images added to the CBIS-film database")
        plt.ylabel(f"Δ {names[metric]}  vs. full-CBIS baseline")
        plt.title(f"Film vs digital augmentation — Δ{names[metric]} at matched dose\n"
                  f"(Ensembler B, paired-151 test, mean±std over 4 seeds)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = os.path.join(config.OUTPUT_DIR, f"compare_film_vs_digital_{metric}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=140)
        plt.close()
        print("saved", out)

    # matched-dose numbers
    for metric in ["auc"]:
        print(f"\n{names[metric]} delta vs full-CBIS baseline:")
        for tag, lab, _ in SERIES:
            x, y, _ = data[tag][metric]
            # peak and endpoint
            k = int(np.argmax(y))
            print(f"  {lab:20s}: peak {y[k]-y[0]:+.4f} @ +{int(x[k])} imgs | "
                  f"full {y[-1]-y[0]:+.4f} @ +{int(x[-1])} imgs")
        # matched ~115 imgs (MIAS max) across all
        print(f"\n  at matched ~115 extra imgs (MIAS's max):")
        for tag, lab, _ in SERIES:
            x, y, _ = data[tag][metric]
            j = int(np.argmin(np.abs(x - 115)))
            print(f"    {lab:20s}: {y[j]-y[0]:+.4f}  (@ +{int(x[j])} imgs)")


if __name__ == "__main__":
    main()
