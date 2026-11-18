"""E5 (reviewer R2.6): parameter-efficient fine-tuning (LoRA) of the two transformer
encoders (DINOv2 ViT-L/14 at 518 and RAD-DINO), decision D5.

Identical to finetune_vit.py (same folds from cache/finetune_fold_assignment.json, same
data pipeline, same new linear head, epochs <= 20, patience 5 on the fold's validation
AUC, class-balanced loss, AdamW wd 1e-4, AMP, batch 1 x 6 accumulation), except that the
backbone weights stay FROZEN and low-rank adapters are trained instead:
  LoRA on every attention query/key/value projection, rank r=8, alpha=16 (scale 2),
  A ~ Kaiming-uniform, B = 0 (so training starts exactly at the frozen model),
  learning rate 1e-4 for the adapters, 1e-3 for the head.
Hyperparameters fixed before running, not tuned.

Fold-matched embeddings stored as dinov2_vitl14_r518_lora_fold{i} / rad_dino_lora_fold{i};
evaluate with finetune_lora_eval.py. Logs wall-clock, peak memory and trainable params.

Run: python finetune_lora.py --model dinov2 | raddino  [--smoke]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import db
import finetune_vit as FV

RANK, ALPHA, LR_LORA, LR_HEAD, WD = 8, 16, 1e-4, 1e-3, 1e-4
PREFIX = {"dinov2": "dinov2_vitl14_r518_lora", "raddino": "rad_dino_lora"}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=RANK, alpha=ALPHA):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def add_lora(backbone):
    targets = []
    for name, mod in backbone.named_modules():
        for child_name, child in mod.named_children():
            if isinstance(child, nn.Linear) and child_name in ("qkv", "query", "key", "value"):
                targets.append((mod, child_name, child))
    for parent, cname, child in targets:
        setattr(parent, cname, LoRALinear(child))
    return len(targets)


def build(model_key):
    model = FV.build_model(model_key)
    for p in model.backbone.parameters():
        p.requires_grad = False
    n_wrapped = add_lora(model.backbone)
    lora = [p for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")]
    n_train = sum(p.numel() for p in lora) + sum(p.numel() for p in model.head.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    return model.to(FV._DEVICE), lora, n_wrapped, n_train, n_total


def train_fold(model_key, fit_rows, val_rows, epochs, patience, smoke, accum=6):
    model, lora, n_wrapped, n_train, n_total = build(model_key)
    print(f"  LoRA on {n_wrapped} projections; trainable {n_train:,} of {n_total:,} "
          f"({100 * n_train / n_total:.2f}%)", flush=True)
    n_pos = sum(1 for r in fit_rows if r[2] == 1)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(fit_rows) - n_pos) / max(n_pos, 1)],
                                                        device=FV._DEVICE))
    opt = torch.optim.AdamW([{"params": lora, "lr": LR_LORA},
                             {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda")
    fit_dl = DataLoader(FV.CbisDataset(fit_rows, True), batch_size=1, shuffle=True, num_workers=2,
                        drop_last=True)
    val_dl = DataLoader(FV.CbisDataset(val_rows, False), batch_size=8, shuffle=False, num_workers=2)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(fit_dl):
            x, y = x.to(FV._DEVICE), y.to(FV._DEVICE)
            with torch.autocast("cuda"):
                loss = crit(model(x)[0], y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        model.eval(); ys, ss = [], []
        with torch.no_grad():
            for x, y, _ in val_dl:
                with torch.autocast("cuda"):
                    ss.append(torch.sigmoid(model(x.to(FV._DEVICE))[0]).float().cpu().numpy())
                ys.append(y.numpy())
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(ss))
        print(f"    epoch {ep + 1}: val AUC {auc:.4f}", flush=True)
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if smoke:
            break
    model.load_state_dict(best_state)
    return model, best, n_train, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(PREFIX))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    conn = db.connect()
    rows = FV.load_rows(conn, split="train")
    fold_of = FV._load_fold_of()
    log, log_path = [], os.path.join(FV.CACHE, f"lora_{a.model}_log.json")
    for V in range(1 if a.smoke else 5):
        fit = [(r["id"], r["file_path"], r["binary_label"]) for r in rows if fold_of[r["id"]] != V]
        val = [(r["id"], r["file_path"], r["binary_label"]) for r in rows if fold_of[r["id"]] == V]
        if a.smoke:
            fit, val = fit[:24], val[:8]
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        print(f"[{a.model} LoRA fold {V}] fit={len(fit)} val={len(val)}", flush=True)
        model, best, n_train, n_total = train_fold(a.model, fit, val, 1 if a.smoke else 20, 5, a.smoke)
        name = f"{PREFIX[a.model]}{'_smoke' if a.smoke else ''}_fold{V}"
        n_emb = FV.embed_all(conn, model, name, batch_size=8, limit=32 if a.smoke else None)
        log.append(dict(fold=V, val_auc=best, wall_clock_sec=time.time() - t0,
                        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, n_embedded=n_emb,
                        trainable_params=n_train, total_params=n_total, embedder_name=name,
                        rank=RANK, alpha=ALPHA, lr_lora=LR_LORA, lr_head=LR_HEAD))
        print(f"[{a.model} LoRA fold {V}] done: val AUC {best:.4f}, "
              f"{log[-1]['wall_clock_sec'] / 60:.1f} min, peak {log[-1]['peak_gpu_gb']:.1f} GB", flush=True)
        del model; torch.cuda.empty_cache()
        if not a.smoke:
            json.dump(log, open(log_path, "w"), indent=2)
    print(f"TOTAL {sum(e['wall_clock_sec'] for e in log) / 3600:.2f} GPU-hours")


if __name__ == "__main__":
    main()
