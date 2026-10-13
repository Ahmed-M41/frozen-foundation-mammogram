"""Phase 2 of the reviewer-response control: fine-tune DINOv2 ViT-L/14 and RAD-DINO
end to end on the same patient-grouped CBIS-DDSM training pool, exactly like
finetune_mammo_clip.py does for Mammo-CLIP. MammoScreen is NOT fine-tuned here --
it is already pre-trained on CBIS-DDSM itself, so further fine-tuning on the same
~985-image pool would not test anything new.

Same discipline as finetune_mammo_clip.py: patient-grouped 5-fold (REUSES the exact
fold assignment in cache/finetune_fold_assignment.json, so all three fine-tuned
encoders share the same folds and stay directly comparable), fold-matched backbones
(fold i's backbone never saw fold i's labels), differential LR (backbone 1e-5, new
head 1e-3), AMP, gradient accumulation.

Preprocessing matches the frozen embedders exactly (embedders.py): no CLAHE
(dinov2_vitl14_r518_noclahe / rad_dino_noclahe are the variants used in the main
pipeline), grayscale image read then RGB-replicated, square-resized to 518x518,
ImageNet mean/std normalization.

Stores fold-matched embeddings as:
  dinov2_vitl14_r518_ft_fold{i}   (dim 1024)
  rad_dino_ft_fold{i}             (dim 768)

Run:  python finetune_vit.py --model dinov2            # full run (all 5 folds)
      python finetune_vit.py --model raddino
      python finetune_vit.py --model dinov2 --smoke    # fast sanity/calibration check
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config
import db

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = os.path.join(config.PROJECT_DIR, "cache")
os.makedirs(CACHE, exist_ok=True)
FOLD_JSON = os.path.join(CACHE, "finetune_fold_assignment.json")

DATASET = "cbis_full"
SIZE = 518
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_SPLITS = 5

MODELS = {
    "dinov2":  dict(embedder_prefix="dinov2_vitl14_r518_ft", dim=1024,
                     log_name="finetune_dinov2_log.json"),
    "raddino": dict(embedder_prefix="rad_dino_ft", dim=768,
                     log_name="finetune_raddino_log.json"),
}


# --- data ---------------------------------------------------------------------

_RESIZE = transforms.Compose([
    transforms.Resize((SIZE, SIZE), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _to_tensor(path, train_aug):
    im = Image.open(path).convert("L").convert("RGB")
    if train_aug and np.random.rand() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return _RESIZE(im)


class CbisDataset(Dataset):
    def __init__(self, rows, train_aug):
        self.rows = rows                    # list of (sample_id, file_path, label)
        self.train_aug = train_aug

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sid, path, label = self.rows[idx]
        x = _to_tensor(path, self.train_aug)
        return x, torch.tensor(float(label)), sid


def load_rows(conn, split=None):
    q = "SELECT id,file_path,binary_label,patient_id FROM samples WHERE dataset=?"
    params = [DATASET]
    if split is not None:
        q += " AND split=?"
        params.append(split)
    return conn.execute(q, params).fetchall()


def _load_fold_of():
    with open(FOLD_JSON) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# --- model ----------------------------------------------------------------------

class DinoFineTuneModel(nn.Module):
    """Pretrained DINOv2 ViT-L/14 backbone (torch.hub, already pretrained) + new
    linear head. Matches embedders._dinov2: model(x) directly returns the pooled
    CLS-style feature (dim 1024), no wrapper needed."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self.dim = 1024
        self.head = nn.Linear(self.dim, 1)

    def forward(self, x):
        pooled = self.backbone(x)
        return self.head(pooled).squeeze(1), pooled


class RadDinoFineTuneModel(nn.Module):
    """Pretrained RAD-DINO backbone (HF, already pretrained) + new linear head.
    Matches embedders._build_hf_vit(pooling='cls'): CLS token of last_hidden_state."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("microsoft/rad-dino")
        self.dim = self.backbone.config.hidden_size
        self.head = nn.Linear(self.dim, 1)

    def forward(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        pooled = h[:, 0]
        return self.head(pooled).squeeze(1), pooled


def build_model(model_key):
    return DinoFineTuneModel() if model_key == "dinov2" else RadDinoFineTuneModel()


# --- train / eval one fold -------------------------------------------------------

def train_fold(model_key, fold_id, fit_rows, val_rows, epochs, batch_size, patience,
                smoke, accum_steps=1, eval_batch_size=8):
    model = build_model(model_key).to(_DEVICE)

    n_pos = sum(1 for r in fit_rows if r[2] == 1)
    n_neg = len(fit_rows) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=_DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=_DEVICE == "cuda")

    fit_dl = DataLoader(CbisDataset(fit_rows, train_aug=True), batch_size=batch_size,
                         shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(CbisDataset(val_rows, train_aug=False), batch_size=eval_batch_size,
                         shuffle=False, num_workers=2)

    best_auc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(fit_dl):
            x, y = x.to(_DEVICE, non_blocking=True), y.to(_DEVICE, non_blocking=True)
            with torch.autocast("cuda", enabled=_DEVICE == "cuda"):
                logit, _ = model(x)
                loss = crit(logit, y) / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        model.eval()
        ys, ss = [], []
        with torch.no_grad():
            for x, y, _ in val_dl:
                x = x.to(_DEVICE, non_blocking=True)
                with torch.autocast("cuda", enabled=_DEVICE == "cuda"):
                    logit, _ = model(x)
                ss.append(torch.sigmoid(logit).float().cpu().numpy())
                ys.append(y.numpy())
        ys, ss = np.concatenate(ys), np.concatenate(ss)
        auc = roc_auc_score(ys, ss) if len(set(ys.tolist())) > 1 else 0.0
        print(f"    fold {fold_id} epoch {epoch+1}/{epochs}: val_auc={auc:.4f}", flush=True)

        if auc > best_auc:
            best_auc, bad_epochs = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"    fold {fold_id}: early stop at epoch {epoch+1}")
                break
        if smoke:
            break

    model.load_state_dict(best_state)
    return model, best_auc


@torch.no_grad()
def embed_all(conn, model, embedder_name, batch_size=8, limit=None):
    rows = load_rows(conn)                     # ALL samples (train+test)
    if limit is not None:
        rows = rows[:limit]
    model.eval()
    dl = DataLoader(CbisDataset([(r["id"], r["file_path"], r["binary_label"]) for r in rows],
                                train_aug=False),
                    batch_size=batch_size, shuffle=False, num_workers=2)
    n = 0
    for x, _y, sids in dl:
        x = x.to(_DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=_DEVICE == "cuda"):
            _logit, pooled = model(x)
        pooled = pooled.float().cpu().numpy()
        for sid, vec in zip(sids.tolist(), pooled):
            db.insert_embedding(conn, sid, embedder_name, vec)
        n += len(sids)
    conn.commit()
    return n


# --- main ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum-steps", type=int, default=6)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    spec = MODELS[args.model]

    conn = db.connect()
    all_train_rows = load_rows(conn, split="train")
    fold_of = _load_fold_of()   # REUSE the exact same folds as finetune_mammo_clip.py

    n_folds = 1 if args.smoke else N_SPLITS
    epochs = 1 if args.smoke else args.epochs
    log_path = os.path.join(CACHE, spec["log_name"])
    log = []
    for fold_id in range(n_folds):
        fit_rows = [(r["id"], r["file_path"], r["binary_label"]) for r in all_train_rows
                    if fold_of[r["id"]] != fold_id]
        val_rows = [(r["id"], r["file_path"], r["binary_label"]) for r in all_train_rows
                    if fold_of[r["id"]] == fold_id]
        if args.smoke:
            fit_rows, val_rows = fit_rows[:24], val_rows[:8]
        print(f"[{args.model} fold {fold_id}] fit={len(fit_rows)} val={len(val_rows)}", flush=True)

        torch.cuda.reset_peak_memory_stats() if _DEVICE == "cuda" else None
        t0 = time.time()
        model, best_auc = train_fold(args.model, fold_id, fit_rows, val_rows, epochs,
                                      args.batch_size, args.patience, args.smoke,
                                      accum_steps=args.accum_steps)
        embedder_name = f"{spec['embedder_prefix']}_fold{fold_id}"
        embed_limit = 32 if args.smoke else None
        n_embedded = embed_all(conn, model, embedder_name, batch_size=8, limit=embed_limit)
        wall = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if _DEVICE == "cuda" else None

        entry = dict(fold=fold_id, val_auc=best_auc, wall_clock_sec=wall,
                     peak_gpu_gb=peak_gb, n_embedded=n_embedded,
                     embedder_name=embedder_name)
        log.append(entry)
        print(f"[{args.model} fold {fold_id}] done: val_auc={best_auc:.4f} wall={wall/60:.1f}min "
              f"peak_gpu={peak_gb:.2f}GB embedded={n_embedded}", flush=True)

        del model
        if _DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nlog written to {log_path}")
    total_min = sum(e["wall_clock_sec"] for e in log) / 60
    print(f"TOTAL wall-clock: {total_min:.1f} min across {len(log)} fold(s)")


if __name__ == "__main__":
    main()
