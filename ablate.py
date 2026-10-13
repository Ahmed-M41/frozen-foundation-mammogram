"""Controlled ablation: measure each block's INDIVIDUAL effect on the fixed
baseline (Mammo-CLIP-B5 + DINOv2-L + RAD-DINO, 4-way multi-view, CBIS paired-151),
so a helpful block can't hide behind a harmful one. Each config is C-swept and the
best-tuned result (by AUC) is reported. Add/remove configs freely.

Run:  python ablate.py
"""
import contextlib
import io

import multiview_probe as MV

BASE = ["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"]
CS = [0.02, 0.05, 0.1, 0.2, 0.5]

CONFIGS = [
    ("baseline: MC-B5 + DINOv2-L + RAD-DINO", BASE),
    ("mammoscreen_feat  ALONE",               ["mammoscreen_feat"]),
    ("baseline + mammoscreen_feat",           BASE + ["mammoscreen_feat"]),
]


def best_of(encs, dataset="cbis_full"):
    best = None
    for C in CS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _full, pm = MV.run(encs, dataset=dataset, mode="fourway", C=C)
        except Exception as e:  # noqa: BLE001
            return {"err": repr(e)[:80]}
        if best is None or pm["auc"] > best["auc"]:
            best = dict(C=C, **pm)
    return best


def main():
    print(f"{'config':44s} {'bestC':>6} {'acc':>7} {'bal_acc':>8} {'auc':>7}  vs-baseline")
    base_auc = None
    for label, encs in CONFIGS:
        b = best_of(encs)
        if "err" in b:
            print(f"{label:44s}  SKIP ({b['err']})")
            continue
        if base_auc is None:
            base_auc = b["auc"]
            delta = " (baseline)"
        else:
            d = b["auc"] - base_auc
            delta = f" {'+' if d >= 0 else ''}{d:+.4f} AUC"
        print(f"{label:44s} {b['C']:>6} {b['accuracy']:.4f}  {b['bal_acc']:.4f}  {b['auc']:.4f}{delta}")


if __name__ == "__main__":
    main()
