"""Overlay the three growth orders (EXP0 cbis-first / EXP1 mias-first / EXP2 mixed)
to separate DATA COMPOSITION from DATA ORDERING.

Also regenerates each run's own combined single-axes figure (the output/combined view:
all three metrics on one axes) straight from its CSV, so re-plotting never needs a re-run.

Comparison figures, one per metric:
  vs. curve position (% of database) — the axis each run was defined on.
  vs. ACTUAL image count            — the matched-budget view. Curves coinciding here
                                      => the curve is PATH-INDEPENDENT and the MIAS
                                      effect is about the data, not about when it was added.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_scaling_order import combined_plot, POINTS

RUNS = [
    ("output/exp0_cbis_first", "cbis_first", "EXP0: CBIS -> MIAS", "#d62728"),
    ("output/exp1_mias_first", "extra_first", "EXP1: MIAS -> CBIS", "#1f77b4"),
    ("output/exp2_mixed",      "mixed",      "EXP2: random mix",   "#2ca02c"),
]
METRICS = [("auc", "AUC"), ("acc", "Accuracy"), ("bal", "Balanced acc")]
LABEL = "MIAS (film)"
OUTDIR = "output/exp_order_compare"


def load(folder, order):
    with open(os.path.join(folder, f"B_mias_{order}.csv")) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = {order: load(folder, order) for folder, order, _, _ in RUNS}
    pts = np.array(POINTS)

    # 1. per-run combined figure, in the single-axes view
    for folder, order, _, _ in RUNS:
        d = data[order]
        agg = {m: (d[f"{m}_mean"], d[f"{m}_std"]) for m, _ in METRICS}
        out = os.path.join(folder, f"B_mias_{order}.png")
        combined_plot(out, pts, agg, order, LABEL, 0.6, "oracle")
        print("saved", out)

    # 2. cross-order overlay, one figure per metric, on both x-axes
    for m, mlabel in METRICS:
        for xkey, xlab, suffix in [
                ("pt", "% of database used   (each run on its own growth order)", "pct"),
                ("n_img", "database size (images actually in the retrieval database)", "nimg")]:
            plt.figure(figsize=(9, 5.5))
            for _, order, lab, col in RUNS:
                d = data[order]
                o = np.argsort(d[xkey])
                x, mu, sd = d[xkey][o], d[f"{m}_mean"][o], d[f"{m}_std"][o]
                plt.plot(x, mu, "-o", ms=4, color=col, label=lab)
                plt.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.13)
            base = data["mixed"][f"{m}_mean"][0]
            plt.axhline(base, color="gray", ls=":", lw=1)
            plt.text(plt.xlim()[1], base, "no database (mammoscreen only) ", color="gray",
                     fontsize=9, ha="right", va="bottom")
            if xkey == "pt":
                plt.axvline(100, color="gray", ls="--", lw=1)
                plt.xticks(pts)
            plt.xlabel(xlab)
            plt.ylabel(mlabel)
            plt.title(f"{mlabel} — data-ORDERING ablation (Ensembler B, w=0.6, oracle threshold)\n"
                      f"CBIS + MIAS, paired-151 test, mean±std over 4 seeds")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="lower right")
            plt.tight_layout()
            out = os.path.join(OUTDIR, f"order_{m}_{suffix}.png")
            plt.savefig(out, dpi=140)
            plt.close()
            print("saved", out)

    # 3. tidy long-format csv of everything plotted
    csv_out = os.path.join(OUTDIR, "order_ablation.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "pt", "n_img"] +
                   [f"{m}_{s}" for m, _ in METRICS for s in ("mean", "std")])
        for _, order, _, _ in RUNS:
            d = data[order]
            for i in range(len(d["pt"])):
                w.writerow([order, int(d["pt"][i]), f"{d['n_img'][i]:.1f}"] +
                           [f"{d[f'{m}_{s}'][i]:.6f}" for m, _ in METRICS for s in ("mean", "std")])
    print("saved", csv_out)

    # 4. matched-budget readout + path-independence test
    print(f"\n{'order':<12} {'pt':>4} {'n_img':>7} {'AUC':>8} {'acc':>8} {'bal':>8}")
    for _, order, _, _ in RUNS:
        d = data[order]
        for i in (0, len(d["pt"]) // 2, len(d["pt"]) - 1):
            print(f"{order:<12} {int(d['pt'][i]):>4} {d['n_img'][i]:>7.0f} "
                  f"{d['auc_mean'][i]:>8.4f} {d['acc_mean'][i]:>8.4f} {d['bal_mean'][i]:>8.4f}")

    lo = max(data[o]["n_img"][data[o]["n_img"] > 0].min() for _, o, _, _ in RUNS)
    hi = min(data[o]["n_img"].max() for _, o, _, _ in RUNS)
    grid = np.linspace(lo, hi, 25)
    print(f"\n--- path-independence: max spread between orders at MATCHED image counts "
          f"[{lo:.0f}..{hi:.0f}] ---")
    for m, mlabel in METRICS:
        cur = []
        for _, order, _, _ in RUNS:
            d = data[order]
            o = np.argsort(d["n_img"])
            cur.append(np.interp(grid, d["n_img"][o], d[f"{m}_mean"][o]))
        spread = np.max(cur, 0) - np.min(cur, 0)
        print(f"  {mlabel:<14} max {spread.max():.4f} at ~{grid[spread.argmax()]:.0f} imgs "
              f"| at >=500 imgs: {spread[grid >= 500].max():.4f}")


if __name__ == "__main__":
    main()
