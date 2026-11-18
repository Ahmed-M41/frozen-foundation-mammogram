"""E4 (reviewer R2.5): an ImageNet-initialised ResNet-50 trained END TO END on the whole
mammogram, as a conventional fully-trained baseline under the identical protocol.

Same patient-grouped folds as the fine-tuning control (cache/finetune_fold_assignment.json),
same early-stopping rule (patience 5 on the fold's validation AUC, at most 20 epochs), same
horizontal-flip-only augmentation, class-balanced loss, AdamW with weight decay 1e-4 and
mixed precision. Input: whole CBIS-DDSM mammogram, grayscale replicated to three channels,
resized to 1024x512 (decision D3), ImageNet normalisation. Learning rate: backbone 1e-4,
head 1e-3 (the 1e-5 backbone rate of the fine-tuning control is for encoders already
pre-trained on medical images; an ImageNet network needs more adaptation; to be confirmed).

Each fold model scores the official test set; scores are averaged over the five fold
models, then over the views of each breast, exactly like the frozen method. Evaluation:
finetune_resnet50_eval.py.

Variant: R50_LR_BACKBONE=1e-5 R50_TAG=_lr1e-5 python finetune_resnet50_e2e.py
Run: python finetune_resnet50_e2e.py   ->  cache/resnet50_e2e_preds.json, cache/resnet50_e2e_log.json
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import config
import db

DEV = "cuda"
CACHE = os.path.join(config.PROJECT_DIR, "cache")
H, W = 1024, 512
MEAN, STD = 0.449, 0.226          # ImageNet mean/std averaged over RGB (grayscale input)
EPOCHS, PATIENCE, BATCH = 20, 5, 8
LR_BACKBONE = float(os.environ.get("R50_LR_BACKBONE", "1e-4"))
LR_HEAD, WD = 1e-3, 1e-4
TAG = os.environ.get("R50_TAG", "")   # output suffix for variants, e.g. "_lr1e-5"


def load_images(rows):
    arr = np.empty((len(rows), H, W), dtype=np.uint8)
    for k, r in enumerate(rows):
        arr[k] = np.asarray(Image.open(r["file_path"]).convert("L").resize((W, H), Image.BILINEAR))
        if k % 200 == 0:
            print(f"  decoded {k}/{len(rows)}", flush=True)
    return arr


class DS(Dataset):
    def __init__(self, imgs, labels, aug):
        self.imgs, self.labels, self.aug = imgs, labels, aug

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        a = self.imgs[i].astype(np.float32) / 255.0
        if self.aug and np.random.rand() < 0.5:
            a = a[:, ::-1].copy()
        a = (a - MEAN) / STD
        return torch.from_numpy(a)[None].repeat(3, 1, 1), torch.tensor(float(self.labels[i]))


def model_new():
    import timm
    return timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=1).to(DEV)


@torch.no_grad()
def predict(model, imgs):
    model.eval()
    out = []
    dl = DataLoader(DS(imgs, np.zeros(len(imgs)), False), batch_size=16, shuffle=False)
    for x, _ in dl:
        with torch.autocast("cuda"):
            out.append(torch.sigmoid(model(x.to(DEV)).squeeze(1)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    torch.manual_seed(0); np.random.seed(0)
    conn = db.connect()
    rows = conn.execute("SELECT id, file_path, binary_label, split FROM samples "
                        "WHERE dataset='cbis_full' ORDER BY id").fetchall()
    fold_of = {int(k): v for k, v in json.load(open(os.path.join(CACHE, "finetune_fold_assignment.json"))).items()}
    imgs = load_images(rows)
    ids = np.array([r["id"] for r in rows]); y = np.array([r["binary_label"] for r in rows])
    is_tr = np.array([r["split"] == "train" for r in rows]); is_te = ~is_tr
    fold = np.array([fold_of.get(int(i), -1) for i in ids])
    test_scores, oof, log = [], {}, []
    for V in range(5):
        fit = is_tr & (fold != V); val = is_tr & (fold == V)
        model = model_new()
        head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
        body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        opt = torch.optim.AdamW([{"params": body, "lr": LR_BACKBONE}, {"params": head, "lr": LR_HEAD}],
                                weight_decay=WD)
        pw = torch.tensor([(y[fit] == 0).sum() / max((y[fit] == 1).sum(), 1)], device=DEV)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        scaler = torch.amp.GradScaler("cuda")
        dl = DataLoader(DS(imgs[fit], y[fit], True), batch_size=BATCH, shuffle=True, drop_last=True)
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        best, best_state, bad = -1.0, None, 0
        for ep in range(EPOCHS):
            model.train()
            for x, t in dl:
                x, t = x.to(DEV, non_blocking=True), t.to(DEV)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    loss = crit(model(x).squeeze(1), t)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            auc = roc_auc_score(y[val], predict(model, imgs[val]))
            print(f"fold {V} epoch {ep + 1}: val AUC {auc:.4f}", flush=True)
            if auc > best:
                best, bad = auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        test_scores.append(predict(model, imgs[is_te]))
        for sid, s in zip(ids[val], predict(model, imgs[val])):
            oof[int(sid)] = float(s)
        log.append(dict(fold=V, val_auc=best, epochs_run=ep + 1, wall_clock_sec=time.time() - t0,
                        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9))
        print(f"fold {V} done: best val AUC {best:.4f}, {log[-1]['wall_clock_sec'] / 60:.1f} min, "
              f"peak {log[-1]['peak_gpu_gb']:.1f} GB", flush=True)
        del model; torch.cuda.empty_cache()
        json.dump(log, open(os.path.join(CACHE, f"resnet50_e2e{TAG}_log.json"), "w"), indent=2)
    te = np.mean(test_scores, 0)
    json.dump(dict(test={int(i): float(s) for i, s in zip(ids[is_te], te)}, oof=oof,
                   config=dict(H=H, W=W, epochs=EPOCHS, patience=PATIENCE, batch=BATCH,
                               lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, wd=WD)),
              open(os.path.join(CACHE, f"resnet50_e2e{TAG}_preds.json"), "w"))
    print(f"TOTAL {sum(e['wall_clock_sec'] for e in log) / 3600:.2f} GPU-hours", flush=True)


if __name__ == "__main__":
    main()
