"""Reviewer-response control experiment: does fine-tuning the pretrained Mammo-CLIP
backbone on the in-domain training set beat freezing it -- and at what cost?

This is NOT "train from scratch" (already answered: worse, see paper). It continues
training the RELEASED Mammo-CLIP checkpoint (same weights `mammo_clip.py` loads
frozen) on the ~985-image CBIS-DDSM train-fold pool, with a new binary head.

Patient-grouped 5-fold, matching the discipline multiview_probe.py's head_cv already
uses one level up (the linear probe). Here it's one level down: for each fold i, a
FRESH copy of the pretrained backbone is fine-tuned excluding fold i, then used to
embed ALL cbis_full images (train+test), stored as embedder_name
f"mammo_clip_b5_ft_fold{i}".

Why per-fold backbones (not one backbone fine-tuned on the full train set): if a
single fine-tuned backbone produced every image's features, its weights would have
been shaped by fold V's labels even when finetune_eval.py later holds fold V out for
the linear probe -- an indirect leak through the encoder. Fold-matched backbones
close that gap: fold i's backbone never saw fold i's labels, so any downstream
linear head evaluated on fold i via fold i's features is honest.

Logs wall-clock + peak GPU memory per fold -- the cost side of the efficiency
argument -- to cache/finetune_log.json. Fold assignment (sample_id -> fold) is
saved to cache/finetune_fold_assignment.json so finetune_eval.py reuses the exact
same split rather than recomputing it.

Run:  python finetune_mammo_clip.py            # full run (all 5 folds)
      python finetune_mammo_clip.py --smoke    # fast sanity check, 1 fold, 1 epoch,
                                                # small image subset
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

import config
import db
import mammo_clip

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = os.path.join(config.PROJECT_DIR, "cache")
os.makedirs(CACHE, exist_ok=True)
FOLD_JSON = os.path.join(CACHE, "finetune_fold_assignment.json")
LOG_JSON = os.path.join(CACHE, "finetune_log.json")

DATASET = "cbis_full"
SIZE = (1520, 912)          # (H, W), same as mammo_clip.py
N_SPLITS = 5
SEED = 42


# --- data ---------------------------------------------------------------------

def _to_tensor(path, train_aug):
    g = Image.open(path).convert("L").resize((SIZE[1], SIZE[0]), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    if train_aug and np.random.rand() < 0.5:
        arr = arr[:, ::-1].copy()          # horizontal flip only; laterality unused
    arr = (arr - mammo_clip.MAMMO_MEAN) / mammo_clip.MAMMO_STD
    return torch.from_numpy(arr)[None].repeat(3, 1, 1)


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


def make_fold_assignment(train_rows, n_splits=N_SPLITS, seed=SEED):
    ids = np.array([r["id"] for r in train_rows])
    y = np.array([r["binary_label"] for r in train_rows])
    pat = np.array([r["patient_id"] for r in train_rows])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = np.full(len(ids), -1, dtype=int)
    for fold_id, (_, va) in enumerate(sgkf.split(ids, y, groups=pat)):
        fold_of[va] = fold_id
    assert (fold_of >= 0).all()
    return {int(i): int(f) for i, f in zip(ids, fold_of)}


# --- model ----------------------------------------------------------------------

class FineTuneModel(nn.Module):
    """Pretrained EfficientNet-B5 backbone (Mammo-CLIP init) + new linear head."""

    def __init__(self, arch="efficientnet-b5"):
        super().__init__()
        from efficientnet_pytorch import EfficientNet
        self.backbone = EfficientNet.from_name(arch)
        self.dim = self.backbone._conv_head.out_channels
        self.head = nn.Linear(self.dim, 1)

    def load_pretrained(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw = ck["model"] if "model" in ck else ck
        weights = {k[len("image_encoder."):]: v for k, v in raw.items()
                   if k.startswith("image_encoder.")}
        missing, unexpected = self.backbone.load_state_dict(weights, strict=False)
        backbone_missing = [m for m in missing if not m.startswith("_fc")]
        assert not backbone_missing, f"missing backbone keys: {backbone_missing[:5]}"
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"

    def forward(self, x):
        fmap = self.backbone.extract_features(x)
        pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        return self.head(pooled).squeeze(1), pooled


# --- train / eval one fold -------------------------------------------------------

def train_fold(fold_id, fit_rows, val_rows, epochs, batch_size, patience, smoke,
               accum_steps=1, eval_batch_size=8):
    model = FineTuneModel().to(_DEVICE)
    model.load_pretrained(mammo_clip.download_ckpt("mammo_clip_b5"))

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
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="physical batch size (fine-tuning a full B5 at 1520x912 "
                         "needs this small on 16GB; use --accum-steps to reach a "
                         "larger effective batch)")
    ap.add_argument("--accum-steps", type=int, default=3,
                    help="gradient accumulation steps (effective batch = "
                         "batch-size * accum-steps)")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 1 epoch, small subset, embeds only the smoke "
                         "subset (not the full dataset) -- sanity check only")
    args = ap.parse_args()

    conn = db.connect()
    all_train_rows = load_rows(conn, split="train")
    fold_of = make_fold_assignment(all_train_rows)
    with open(FOLD_JSON, "w") as f:
        json.dump(fold_of, f)
    print(f"fold assignment saved to {FOLD_JSON} "
          f"({len(fold_of)} train images, {N_SPLITS} folds)", flush=True)

    n_folds = 1 if args.smoke else N_SPLITS
    epochs = 1 if args.smoke else args.epochs
    log = []
    for fold_id in range(n_folds):
        fit_rows = [(r["id"], r["file_path"], r["binary_label"]) for r in all_train_rows
                    if fold_of[r["id"]] != fold_id]
        val_rows = [(r["id"], r["file_path"], r["binary_label"]) for r in all_train_rows
                    if fold_of[r["id"]] == fold_id]
        if args.smoke:
            fit_rows, val_rows = fit_rows[:24], val_rows[:8]
        print(f"[fold {fold_id}] fit={len(fit_rows)} val={len(val_rows)}", flush=True)

        torch.cuda.reset_peak_memory_stats() if _DEVICE == "cuda" else None
        t0 = time.time()
        model, best_auc = train_fold(fold_id, fit_rows, val_rows, epochs,
                                     args.batch_size, args.patience, args.smoke,
                                     accum_steps=args.accum_steps)
        embedder_name = f"mammo_clip_b5_ft_fold{fold_id}"
        embed_limit = 32 if args.smoke else None
        n_embedded = embed_all(conn, model, embedder_name, batch_size=8, limit=embed_limit)
        wall = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if _DEVICE == "cuda" else None

        entry = dict(fold=fold_id, val_auc=best_auc, wall_clock_sec=wall,
                    peak_gpu_gb=peak_gb, n_embedded=n_embedded,
                    embedder_name=embedder_name)
        log.append(entry)
        print(f"[fold {fold_id}] done: val_auc={best_auc:.4f} wall={wall/60:.1f}min "
              f"peak_gpu={peak_gb:.2f}GB embedded={n_embedded}")

        del model
        if _DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open(LOG_JSON, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nlog written to {LOG_JSON}")
    total_min = sum(e["wall_clock_sec"] for e in log) / 60
    print(f"TOTAL wall-clock: {total_min:.1f} min across {len(log)} fold(s)")


if __name__ == "__main__":
    main()
