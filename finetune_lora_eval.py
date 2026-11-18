"""Evaluate LoRA (finetune_lora.py) inside the identical four-extractor classifier used for
the fine-tuning table: LoRA-adapted DINOv2 + LoRA-adapted RAD-DINO (fold-matched) with the
frozen Mammo-CLIP and frozen MammoScreen, C=0.5, threshold 0.5, same folds. Paired
case-level bootstrap CI (5,000) on the AUC difference against the all-frozen row.

Run: python finetune_lora_eval.py  ->  cache/lora_eval.json
"""
import json
import os

import config
import finetune_eval_full4 as F4
import finetune_eval_phase2 as P2
from finetune_paired_bootstrap import _cf, boot


def main():
    P2.FT_PREFIXES = ["dinov2_vitl14_r518_lora", "rad_dino_lora"]
    P2.FROZEN_ENCODERS = ["mammo_clip_b5", "mammoscreen_feat"]
    frozen, pf = _cf(F4.run, C=0.5, mc_fmt="mammo_clip_b5", label="Frozen 4")
    lora, _ = _cf(P2.run, C=0.5)
    common = set(frozen) & set(lora)
    out = {"full": boot(frozen, lora, common), "paired": boot(frozen, lora, common & pf)}
    logs = {}
    for m in ("dinov2", "raddino"):
        p = os.path.join(config.PROJECT_DIR, "cache", f"lora_{m}_log.json")
        if os.path.exists(p):
            lg = json.load(open(p))
            logs[m] = dict(gpu_hours=sum(e["wall_clock_sec"] for e in lg) / 3600,
                           peak_gpu_gb=max(e["peak_gpu_gb"] for e in lg),
                           trainable_params=lg[0]["trainable_params"], total_params=lg[0]["total_params"])
    out["cost"] = logs
    for pop in ("full", "paired"):
        r = out[pop]
        print(f"LoRA {pop:6s} n={r['n']} frozen {r['auc_frozen']:.4f} LoRA {r['auc_ft']:.4f} "
              f"diff {r['delta']:+.4f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
    print("cost:", logs)
    json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", "lora_eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
