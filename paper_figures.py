"""Generate the IEEE Access figure set into paper_ieee_access/figures/.

All numbers come from paper_numbers.json (w=0.6, honest threshold) and the
extend-curve CSVs. Palette: Okabe-Ito (colour-blind safe); confusion matrices use
a single-hue sequential (Blues). Light background (print).
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

FIG = os.path.join(config.PROJECT_DIR, "paper_ieee_access", "figures")
os.makedirs(FIG, exist_ok=True)
# Okabe-Ito (CVD-safe)
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8a8a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})

# BASE4 early-fusion numbers (base4_numbers.py). Nested: NUM[dataset][config].
NUM = json.load(open(os.path.join(config.PROJECT_DIR, "cache", "base4_numbers.json")))
METHOD = "base4"          # the paper's method: BASE4 early feature-fusion

# Single representative-seed confusion counts (scanner_confound_check-style honest
# single-evaluation numbers, NOT summed over the four seeds like NUM's tn/fp/fn/tp,
# so the figure's total matches a real evaluated case count).
CM_SINGLE = json.load(open(os.path.join(config.PROJECT_DIR, "cache", "confusion_singleseed.json")))


def _ortho(ax, pts, c="#555", arrow=True, lw=1.6, z=3, ms=13):
    """Orthogonal (right-angle) connector through pts; arrowhead only at the last point.
    Right-angle routing on a shared spine reads as coordinated and professional."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as _P
    codes = [_P.MOVETO] + [_P.LINETO] * (len(pts) - 1)
    ax.add_patch(FancyArrowPatch(path=_P(pts, codes),
                                 arrowstyle=("-|>" if arrow else "-"),
                                 mutation_scale=ms, lw=lw, color=c,
                                 joinstyle="round", capstyle="round", zorder=z))


def knn_fails():
    """Negative result: kNN retrieval vs 3-encoder classifier vs BASE4 early fusion on CBIS."""
    p3 = NUM["cbis_bm"]["probe3"]["auc"]
    b4 = NUM["cbis_bm"]["base4"]["auc"]
    labels = ["kNN retrieval\n(best of 3 encoders)", "Linear classifier\n(3 encoders)",
              "Classifier (BASE4)\n3 encoders + 1 extractor"]
    vals = [0.707, p3, b4]
    cols = [GREY, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color=GREY, lw=1, ls=":")
    ax.set_ylabel("AUC (CBIS-DDSM, paired)")
    ax.set_title("Retrieval fails, a linear classifier succeeds\non frozen mammography features", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_knn_fails.png"), bbox_inches="tight")
    plt.close(fig); print("saved fig_knn_fails.png")


def early_vs_late():
    """Central comparison: 3-encoder classifier vs late score-fusion vs BASE4 early fusion,
    on CBIS-DDSM (in-domain) and Mini-DDSM (independent). Early fusion is robust on both;
    late fusion helps CBIS but degrades the strong Mini-DDSM classifier."""
    cfgs = [("probe3", "Classifier\n(3 encoders)", BLUE),
            ("late3", "Late score-fusion\n($w{=}0.6$)", ORANGE),
            ("base4", "Early fusion\n(BASE4)", GREEN)]
    datasets = [("cbis_bm", "CBIS-DDSM (B/M)"), ("miniddsm_nc", "Mini-DDSM (N/C)")]
    x = np.arange(len(datasets)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for j, (cfg, lab, col) in enumerate(cfgs):
        vals = [NUM[d][cfg]["auc"] for d, _ in datasets]
        sd = [NUM[d][cfg]["std"]["auc"] for d, _ in datasets]
        bars = ax.bar(x + (j-1)*w, vals, w, yerr=sd, capsize=3, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.006, f"{v:.3f}",
                    ha="center", fontsize=8.5)
    ax.set_xticks(x, [c for _, c in datasets])
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title("Early feature-fusion is robust on both datasets;\nlate score-fusion "
                 "degrades the strong Mini-DDSM classifier", fontsize=10.5)
    ax.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.9,
              edgecolor="none", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_early_vs_late.png"), bbox_inches="tight")
    plt.close(fig); print("saved fig_early_vs_late.png")


def pipeline_diagram():
    """Clean horizontal pipeline schematic; embedders and the one classifier are marked distinctly.
    Early-fusion design: MammoScreen contributes its FEATURES as a 4th frozen encoder, and a single
    multi-view linear classifier on the four concatenated feature sets outputs the decision (no score fusion)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    PURPLE = "#7b52ab"
    fig, ax = plt.subplots(figsize=(10.5, 4.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    def box(x, y, w, h, text, fc, ec, fs=10, z=2, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                    fc=fc, ec=ec, lw=1.6, ls=ls, zorder=z))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=z + 1)
    def arrow(x0, y0, x1, y1, c="#555"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     lw=1.6, color=c, zorder=3))
    ax.text(50, 44, "Frozen models used as-is  $+$  one trained linear classifier",
            ha="center", fontsize=12, weight="bold")
    # --- input / preprocessing ---
    box(2, 17, 12, 10, "CC + MLO\nmammogram", "#eaf3fb", BLUE)
    box(18, 17, 13, 10, "Grayscale\nmammogram", "#fff7e6", ORANGE)
    # --- encoders: dashed GREEN container groups the three NATIVE encoders only ---
    ax.add_patch(FancyBboxPatch((33, 15), 19, 24, boxstyle="round,pad=0.6,rounding_size=2",
                                fc="#f4faf4", ec=GREEN, lw=1.2, ls="--", zorder=0))
    ax.text(42.5, 40.6, "Frozen encoders", ha="center", fontsize=9.5, weight="bold", color=GREEN)
    box(35, 31.5, 15, 6, "Mammo-CLIP", "#eef6ee", GREEN)
    box(35, 24.5, 15, 6, "DINOv2", "#eef6ee", GREEN)
    box(35, 17.5, 15, 6, "RAD-DINO", "#eef6ee", GREEN)
    # --- MammoScreen: a full classifier used as a feature extractor -> dotted PURPLE, OUTSIDE the group ---
    ax.text(42.5, 12.3, "full model, head removed", ha="center", fontsize=7.5,
            style="italic", color=PURPLE)
    box(35, 4.5, 15, 6, "MammoScreen", "#f3eefc", PURPLE, fs=9.5, ls=":")
    # --- the single trained classifier ---
    box(57, 16, 15, 10, "Multi-view\nlinear classifier", "#f3eefc", PURPLE)
    ax.text(64.5, 13.3, "trained in this work", ha="center", fontsize=8,
            style="italic", color=PURPLE)
    box(80, 15.5, 13, 11, "Benign /\nMalignant", "#f0e9f7", PURPLE)
    # --- arrows: orthogonal bus routing (right angles on a shared spine) ---
    box_ys = (34.5, 27.5, 20.5, 7.5)
    _ortho(ax, [(14, 22), (18, 22)])                         # input -> crop
    for by in box_ys:                                        # crop -> extractors (comb out, spine x=33)
        _ortho(ax, [(31, 22), (33, 22), (33, by), (35, by)])
    for by in box_ys:                                        # extractors -> classifier (branches to spine x=55)
        _ortho(ax, [(50, by), (55, by), (55, 21)], arrow=False)
    _ortho(ax, [(55, 21), (57, 21)])                         # spine -> classifier (single arrowhead)
    _ortho(ax, [(72, 21), (80, 21)])                         # classifier -> decision
    # --- legend key: encoder vs model-as-extractor vs full model, stated plainly ---
    def key(y, fc, ec, ls, txt):
        ax.add_patch(FancyBboxPatch((1.4, y), 3.0, 2.0, boxstyle="round,pad=0.2,rounding_size=1",
                                    fc=fc, ec=ec, lw=1.2, ls=ls))
        ax.text(5.6, y + 1.0, txt, ha="left", va="center", fontsize=7.7)
    key(11.2, "#eef6ee", GREEN, "-", "Encoder $\\rightarrow$ feature vector")
    key(6.6, "#f3eefc", PURPLE, ":", "Full model, head removed $\\rightarrow$ vector")
    key(2.0, "#f3eefc", PURPLE, "-", "Full model / classifier $\\rightarrow$ probability")
    fig.savefig(os.path.join(FIG, "fig_pipeline.png"), bbox_inches="tight", dpi=200)
    plt.close(fig); print("saved fig_pipeline.png")


def mvlp_diagram():
    """Architecture of the multi-view linear classifier (the only trained component).
    Per-view 7,680-d feature (four L2-normalised blocks) -> three logistic heads
    (CC specialist, view-agnostic, MLO specialist) fitted with patient-grouped 5-fold
    OOF -> four branch scores averaged per breast into the case score R.
    The view-agnostic head sits between the two specialists (not first) so both its
    incoming connectors stay short and never run past the other heads. All connector
    lines share one dark-grey (not pure black) so none reads as de-emphasised; box
    outlines keep their colour coding. Font sizes are tuned large because this figure
    prints at one column width, which shrinks it roughly 3x from its native canvas.
    Note for anyone editing coordinates here: FancyBboxPatch's boxstyle "pad" inflates
    the drawn outline outward by that amount beyond the given (x, y, w, h) -- a box's
    true visible edge is w/h + 2*pad, not w/h. Text inside a box still centres on the
    nominal w/h (so pad is a free safety margin there), but anything positioned
    OUTSIDE a box relative to its nominal edge must add pad or it will clip into the
    outline -- this bit both the CC/MLO branch labels here previously."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    PURPLE, LINE = "#7b52ab", "#555555"
    fig, ax = plt.subplots(figsize=(10.5, 5.0)); ax.set_xlim(0, 100); ax.set_ylim(4, 48); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=12, z=2, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=1.6, ls=ls, zorder=z))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=z+1)

    def arrow(x0, y0, x1, y1, c=LINE, lw=1.7):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     lw=lw, color=c, zorder=3))

    ax.text(50, 47.5, "Multi-view linear classifier: the only component trained in this work",
            ha="center", fontsize=15.5, weight="bold")

    # --- input composition: one per-view feature = four concatenated blocks ---
    blocks = [("Mammo-CLIP", 2048, "#009E73"), ("DINOv2", 1024, "#009E73"),
              ("RAD-DINO", 768, "#009E73"), ("MammoScreen", 3840, PURPLE)]
    total = sum(b[1] for b in blocks)
    x0, ybot, H, wseg = 3.0, 19.0, 18.0, 5.0
    yy = ybot
    for name, dim, col in blocks:
        hseg = H * dim / total
        ax.add_patch(FancyBboxPatch((x0, yy), wseg, hseg, boxstyle="square,pad=0",
                                    fc=col, ec="white", lw=1.2, alpha=0.85, zorder=2))
        yy += hseg
    # group labels: three encoders (bottom half) vs MammoScreen head-removed (top half).
    # Pulled apart from the "$L_2$-normalised" caption below (extra headroom at ybot-3.6)
    # so bigger text doesn't collide with it.
    ax.text(x0 + wseg + 0.4, ybot + H*0.22, "3 encoders\n3,840-d", ha="left",
            va="center", fontsize=9, color="#2a7a5a")
    ax.text(x0 + wseg + 0.4, ybot + H*0.84, "MammoScreen\n3,840-d", ha="left",
            va="center", fontsize=9, color=PURPLE)
    ax.text(x0 + wseg/2, ybot + H + 2.0, "Per-view feature", ha="center", fontsize=11.5, weight="bold")
    ax.text(x0 + wseg/2, ybot - 3.6, "$L_2$-normalised,\nconcatenated (7,680-d)", ha="center",
            fontsize=9.5, style="italic", color="#444")

    # --- two views of one breast ---
    box(22, 32, 12, 6.5, "CC view\nfeature", "#eaf3fb", "#0072B2", fs=12.5)
    box(22, 11.5, 12, 6.5, "MLO view\nfeature", "#eaf3fb", "#0072B2", fs=12.5)
    # feature stack -> the two views (comb out, spine x=20.6, clear of the block labels)
    _ortho(ax, [(8, 28), (20.6, 28), (20.6, 35.25), (22, 35.25)], c=LINE)
    _ortho(ax, [(8, 28), (20.6, 28), (20.6, 14.75), (22, 14.75)], c=LINE)

    # --- three logistic heads: CC specialist (top), view-agnostic (middle), MLO specialist
    # (bottom). Putting the view-agnostic head second means both its connectors run only to
    # the mid-line, instead of one spanning the full height past the other head's box. ---
    box(43, 36, 17, 6.5, "CC specialist\nlogistic head", "#f3eefc", PURPLE, fs=12.5)
    box(43, 21.75, 17, 6.5, "View-agnostic\nlogistic head", "#f3eefc", PURPLE, fs=12.5)
    box(43, 8, 17, 6.5, "MLO specialist\nlogistic head", "#f3eefc", PURPLE, fs=12.5)

    # views -> heads: both lines out of a view box break at the SAME x (=39) and ride one
    # shared vertical trunk (CC's pair and MLO's pair together span the trunk's full
    # 11.25-39.25 height), each peeling off horizontally only at its target's height.
    # The two "cross" lines into View-agnostic peel off at slightly different heights
    # (25.8 / 24.2) so they still show as two distinct arrowheads, not one merged point.
    _ortho(ax, [(34, 35.25), (39, 35.25), (39, 39.25), (43, 39.25)], c=LINE)  # CC  -> CC spec.
    _ortho(ax, [(34, 35.25), (39, 35.25), (39, 25.8), (43, 25.8)], c=LINE)    # CC  -> agnostic
    _ortho(ax, [(34, 14.75), (39, 14.75), (39, 24.2), (43, 24.2)], c=LINE)    # MLO -> agnostic
    _ortho(ax, [(34, 14.75), (39, 14.75), (39, 11.25), (43, 11.25)], c=LINE)  # MLO -> MLO spec.

    # --- heads -> mean -> R (view-agnostic contributes two branches; comb in, spine x=67).
    # Head boxes are drawn with boxstyle pad=0.6, so their true right edge is at
    # 43+17+0.6=60.6, not the nominal 60 -- the CC/MLO/"CC & MLO" labels below are kept
    # clear of that true edge, not the nominal one, which is what clipped them before. ---
    box(70, 20.5, 12, 9, "mean of the\nfour branch\nscores", "#fff7e6", "#E69F00", fs=11)
    for hy in (39.25, 25.0, 11.25):
        _ortho(ax, [(60, hy), (67, hy), (67, 25)], c=LINE, arrow=False)
    ax.text(63.8, 40.25, "CC", ha="center", fontsize=10, color="#333")
    ax.text(63.8, 27.3, "CC &\nMLO", ha="center", fontsize=10, color="#333", linespacing=1.15)
    ax.text(63.8, 12.25, "MLO", ha="center", fontsize=10, color="#333")
    _ortho(ax, [(67, 25), (70, 25)], c=LINE)
    box(86, 20.5, 13, 9, "Breast\nscore $R$", "#f0e9f7", PURPLE, fs=13)
    _ortho(ax, [(82, 25), (86, 25)], c=LINE)
    ax.text(84.5, 16.8, "one score per breast (case level)", ha="center", fontsize=9.5,
            style="italic", color="#444")
    fig.savefig(os.path.join(FIG, "fig_mvlp.png"), bbox_inches="tight", dpi=200)
    plt.close(fig); print("saved fig_mvlp.png")


def mvlp_training_diagram():
    """How the only trained component is fitted: patient-grouped 5-fold cross-validation.
    Shows (i) the split unit is the patient, not the image (a breast's CC+MLO never cross
    the boundary); (ii) the five folds are used two ways -- held-out slices give unbiased
    out-of-fold training scores (-> honest threshold), and every fold model also scores the
    untouched official test set, the five then averaged into the headline test scores."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    PURPLE, INK = "#7b52ab", "#333"
    FITC, HOLDC = "#8fbfe0", "#c9c9c9"
    fig, ax = plt.subplots(figsize=(10.5, 5.4)); ax.set_xlim(0, 100); ax.set_ylim(3, 50); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9.0, z=4, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=1.5, ls=ls, zorder=z))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=z+1)

    ax.text(50, 48.2, "Training the classifier: patient-grouped, out-of-fold cross-validation",
            ha="center", fontsize=12, weight="bold")
    # (the "CC/MLO never cross the boundary" guarantee and the "identical protocol for all
    # three heads" detail both live in the caption/body text, not the image -- see paper.tex)

    # --- the 5-fold grid: rows = folds, 5 patient groups per row ---
    gx, gw, ncol = 7.0, 40.0, 5
    cw = gw / ncol
    top, rh, gap = 38.0, 4.0, 0.9
    ax.text(gx + gw/2, top + 2.6, "Training set  $\\rightarrow$  5 stratified patient groups",
            ha="center", fontsize=8.4, weight="bold", color=INK)
    for i in range(5):
        yb = top - i * (rh + gap) - rh
        ax.text(gx - 1.4, yb + rh/2, f"Fold {i+1}", ha="right", va="center", fontsize=8.2, color=INK)
        for j in range(ncol):
            held = (j == i)
            ax.add_patch(Rectangle((gx + j*cw, yb), cw, rh, fc=(HOLDC if held else FITC),
                                   ec="white", lw=1.3, hatch=("////" if held else None), zorder=3))
        # per-fold outward arrow stub on the right of the grid
    # legend for the grid
    ax.add_patch(Rectangle((gx, 4.8), 2.4, 2.4, fc=FITC, ec="white"))
    ax.text(gx + 3.0, 6.0, "fit set (4 groups)", va="center", fontsize=7.6, color=INK)
    ax.add_patch(Rectangle((gx + 17, 4.8), 2.4, 2.4, fc=HOLDC, ec="white", hatch="////"))
    ax.text(gx + 20.0, 6.0, "held-out group (out-of-fold)", va="center", fontsize=7.6, color=INK)

    gridR = gx + gw   # right edge of grid
    gridB = top - 4 * (rh + gap) - rh   # bottom edge

    # --- USE 1: held-out slices -> out-of-fold training scores -> threshold ---
    _ortho(ax, [(gx + gw/2, gridB - 0.3), (gx + gw/2, 13.2)], c="#666")
    box(3, 7.6, 48, 5.4,
        "Out-of-fold training scores  $\\rightarrow$\noperating threshold + training AUC (unbiased)",
        "#eef6ee", GREEN, fs=7.9)

    # --- USE 2: the five fitted fold models score the untouched test set, then average ---
    box(60, 33.5, 34, 5.2, "Official test set\n(never in any fold)", "#f2f2f2", "#8a8a8a", fs=8.4)
    box(66, 20.8, 22, 6.2, "Average of the\n5 fold models", "#fff7e6", ORANGE, fs=8.6)
    box(60, 9.7, 34, 4.6, "Test scores  $\\rightarrow$  accuracy, AUC (headline)", "#f0e9f7", PURPLE, fs=8.2)
    # grid -> average node (the five fitted models)
    _ortho(ax, [(gridR + 0.4, top - rh/2 - 1), (55, top - rh/2 - 1), (55, 23.9), (66, 23.9)], c="#666")
    ax.text(58.5, 30.5, "5 fitted\nfold models", ha="center", fontsize=7.2, color="#666")
    # test set -> average node (equal 6.5-unit gaps above and below the average box)
    _ortho(ax, [(77, 33.5), (77, 27.0)], c="#666")
    # average -> test scores
    _ortho(ax, [(77, 20.8), (77, 14.3)], c="#666")

    fig.savefig(os.path.join(FIG, "fig_mvlp_training.png"), bbox_inches="tight", dpi=200)
    plt.close(fig); print("saved fig_mvlp_training.png")


def cost_diagram():
    """The economy of the method: ~480M frozen parameters vs a 115k-parameter trained
    classifier (log scale), and the CPU-only wall-clock to fit that classifier. Numbers measured by
    benchmark_mvlp.py (cache/mvlp_cost.json); frozen counts are published architecture sizes."""
    PURPLE = "#7b52ab"
    cost = json.load(open(os.path.join(config.PROJECT_DIR, "cache", "mvlp_cost.json")))
    trained = cost["cbis"]["params_head_cv"]                 # 115,215 (same for both)
    frozen_parts = [("EfficientNet-B5", 30e6), ("DINOv2 ViT-L/14", 300e6),
                    ("RAD-DINO ViT-B/14", 86e6), ("3$\\times$EffNetV2-S", 64e6)]
    frozen = sum(v for _, v in frozen_parts)                 # ~480M

    fig, (axp, axt) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    # --- Panel A: parameters, log scale ---
    axp.barh([1], [frozen], color=GREEN, height=0.55, zorder=3)
    axp.barh([0], [trained], color=PURPLE, height=0.55, zorder=3)
    axp.set_yticks([0, 1], ["Trained classifier\n(this work)", "Frozen backbones\n(used as-is)"])
    axp.set_xscale("log"); axp.set_xlim(1e4, 2e9)
    axp.text(frozen*1.6, 1, "$\\approx$480 M", va="center", fontsize=10, weight="bold", color=GREEN)
    axp.text(trained*1.6, 0, f"{trained:,}\n($\\approx$0.02% of all params)", va="center",
             fontsize=9, weight="bold", color=PURPLE)
    axp.set_xlabel("Parameters (log scale)")
    axp.set_title("The only trained part is $\\sim$4,000$\\times$ smaller\nthan the frozen "
                  "backbones it reads", fontsize=10)
    axp.grid(axis="x", which="both", alpha=0.25)

    # --- Panel B: CPU fit time ---
    ds = [("CBIS-DDSM", cost["cbis"]["head_cv"]["mean"]),
          ("Mini-DDSM", cost["mini"]["head_cv"]["mean"])]
    axt.bar([0, 1], [t for _, t in ds], color=[BLUE, ORANGE], width=0.6, zorder=3)
    for i, (_, t) in enumerate(ds):
        axt.text(i, t + 0.4, f"{t:.1f}s", ha="center", fontsize=10, weight="bold")
    axt.set_xticks([0, 1], [d for d, _ in ds])
    axt.set_ylabel("Fit time (s)"); axt.set_ylim(0, 20)
    axt.set_title("Classifier fits in seconds\non a single CPU (no GPU)", fontsize=10)
    axt.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_cost.png"), bbox_inches="tight", dpi=200)
    plt.close(fig); print("saved fig_cost.png")


def _sample_grid(cells, fname, dataset):
    """2x2 view/pathology grid: rows = Benign/Malignant case, cols = CC/MLO.

    cells[row][col] is an image path. Real pipeline images (breast region,
    grayscale; no contrast enhancement, matching the method). ``dataset`` is
    printed as a title so each panel is self-identifying.
    """
    import matplotlib.image as mpimg
    rowlabels = ["Benign", "Malignant"]
    collabels = ["CC", "MLO"]
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 6.25))
    for i in range(2):
        for j in range(2):
            ax = axes[i][j]
            ax.imshow(mpimg.imread(cells[i][j]), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#333"); s.set_linewidth(0.8)
            if i == 0:
                ax.set_title(collabels[j], fontsize=12, fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(rowlabels[i], fontsize=12, fontweight="bold", labelpad=8)
    fig.suptitle(dataset, fontsize=15, fontweight="bold", y=0.985)
    plt.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.02, wspace=0.05, hspace=0.06)
    fig.savefig(os.path.join(FIG, fname), dpi=200)
    plt.close(fig); print("saved", fname)


def sample_panel():
    """Two dataset-illustration figures, one per dataset. Each is a 2x2 grid:
    a benign case (top) and a malignant case (bottom), each shown in both the
    CC (left) and MLO (right) views. Hand-vetted, same breast per case, hard-
    coded so the figures are deterministic and reviewer-stable.
    """
    cbis = config.CBIS_PREPROC_DIR
    mini = os.path.join(config.DATA_ROOT, "preproc_jpg_miniddsm")
    # CBIS-DDSM: benign case P_00670 RIGHT, malignant case P_00695 RIGHT
    _sample_grid([
        [os.path.join(cbis, "1-053-3c02527ba6.jpg"), os.path.join(cbis, "1-001-720369c991.jpg")],
        [os.path.join(cbis, "1-001-4907cc5885.jpg"), os.path.join(cbis, "1-164-2b94238f37.jpg")],
    ], "fig_samples_cbis.png", "CBIS-DDSM")
    # Mini-DDSM: benign case A_1264_1 RIGHT, malignant case A_1488_1 LEFT
    _sample_grid([
        [os.path.join(mini, "A_1264_1_RIGHT_CC.jpg"), os.path.join(mini, "A_1264_1_RIGHT_MLO.jpg")],
        [os.path.join(mini, "A_1488_1_LEFT_CC.jpg"), os.path.join(mini, "A_1488_1_LEFT_MLO.jpg")],
    ], "fig_samples_miniddsm.png", "Mini-DDSM")


def confusion(key, neg, pos, title, fname):
    d = CM_SINGLE[key]
    cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]], float)
    rown = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i,j])}\n({rown[i,j]*100:.1f}%)", ha="center",
                    va="center", fontsize=13,
                    color="white" if rown[i, j] > 0.5 else "#222")
    ax.set_xticks([0, 1], [f"pred\n{neg}", f"pred\n{pos}"])
    ax.set_yticks([0, 1], [f"true {neg}", f"true {pos}"], rotation=90, va="center")
    ax.set_title(f"{title}  (n={d['n']})\nAUC {d['auc']:.3f} · bal.acc {d['bal_acc']:.3f}",
                 fontsize=10.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 2), minor=True)
    ax.set_yticks(np.arange(-.5, 2), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))     # fixed figsize, no bbox_inches="tight":
    plt.close(fig)                             # keeps both confusion matrices the same
    print("saved", fname)                      # pixel size regardless of label width


def curve_from_csv(path, title, xlabel, fname, vline=100):
    """Single-axes AUC/Acc/Bal curve (the preferred view) from a scaling CSV."""
    rows = list(csv.DictReader(open(path)))
    pts = sorted({int(float(r["pt"])) for r in rows})
    def agg(metric):
        mean = [np.mean([float(r[metric]) for r in rows if int(float(r["pt"])) == p]) for p in pts]
        sd = [np.std([float(r[metric]) for r in rows if int(float(r["pt"])) == p]) for p in pts]
        return np.array(mean), np.array(sd)
    fig, ax = plt.subplots(figsize=(8.2, 5))
    for metric, lab, col in [("auc", "AUC", BLUE), ("acc", "Accuracy", GREEN),
                             ("bal", "Balanced acc", ORANGE)]:
        mean, sd = agg(metric)
        ax.plot(pts, mean, "-o", ms=4, color=col, label=lab)
        ax.fill_between(pts, mean-sd, mean+sd, color=col, alpha=0.13)
    if vline:
        ax.axvline(vline, color=GREY, ls="--", lw=1)
    ax.set_xlabel(xlabel); ax.set_ylabel("score"); ax.set_title(title, fontsize=10.5)
    ax.set_xticks(pts[::2]); ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    confusion("cbis_bm", "benign", "malignant",
              "CBIS-DDSM  (benign vs malignant, paired)", "cm_cbis_bm.png")
    confusion("miniddsm_nc", "normal", "cancer",
              "Mini-DDSM  (normal vs cancer)", "cm_miniddsm_nc.png")
    knn_fails()
    early_vs_late()
    mvlp_diagram()
    mvlp_training_diagram()
    cost_diagram()
    pipeline_diagram()
    sample_panel()
    # BASE4 database-scaling curves (base4_scaling.py). 10-100% = target pool; 110-200% = + MIAS.
    cbp = os.path.join(config.PROJECT_DIR, "cache", "base4_scaling_cbis.csv")
    if os.path.exists(cbp):
        curve_from_csv(cbp,
                       "CBIS-DDSM performance as the labelled database grows\n"
                       "(early-fusion classifier, 4 seeds)",
                       "% of CBIS-DDSM training pool used", "fig_cbis_mias_curve.png",
                       vline=None)
    else:
        print("[pending] CBIS BASE4 scaling CSV not ready yet:", cbp)
    ncp = os.path.join(config.PROJECT_DIR, "cache", "base4_scaling_mini.csv")
    if os.path.exists(ncp):
        curve_from_csv(ncp,
                       "Mini-DDSM normal/cancer as the labelled database grows\n"
                       "(early-fusion classifier, 4 seeds)",
                       "% of Mini-DDSM training pool used", "fig_miniddsm_nc_curve.png",
                       vline=None)
    else:
        print("[pending] Mini-DDSM BASE4 scaling CSV not ready yet:", ncp)
    print("FIGURES_DONE")
