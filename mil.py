"""Family 1, Option 1: gated attention-MIL head over FROZEN patch embeddings.

The backbone is never trained. We only train a small attention head (Ilse et al.
2018) that learns which patches matter, turning each image's bag of patch
embeddings into one malignancy score. Honest protocol: 5-fold patient-grouped CV
on train (out-of-fold scores give an unbiased train signal for the threshold; the
5 fold-models are averaged on the official test set). Per-image scores are fused
per breast case with our existing CC/MLO fusion, and the threshold is tuned on the
train out-of-fold cases only. Case-level metrics reported (full + paired subset).

Run:  python mil.py --embedder dinov2_vitb14_noclahe --method mean
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import config
import db
import evaluate
from patch_embed import CACHE_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GatedAttentionMIL(nn.Module):
    def __init__(self, D, H=128, dropout=0.4):
        super().__init__()
        self.V = nn.Linear(D, H)
        self.U = nn.Linear(D, H)
        self.w = nn.Linear(H, 1)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(D, 1)

    def rep(self, bag):                             # attention-pooled bag rep [B,D]
        A = self.w(torch.tanh(self.V(bag)) * torch.sigmoid(self.U(bag)))  # [B,P,1]
        A = torch.softmax(A, dim=1)
        return (A * bag).sum(dim=1)

    def forward(self, bag):                         # bag [B,P,D]
        return self.cls(self.drop(self.rep(bag))).squeeze(-1)   # [B]


def _l2(x):                                          # per-patch L2 norm
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def load(encoder, dataset="cbis"):
    d = np.load(os.path.join(CACHE_DIR, f"{encoder}__{dataset}.npz"))
    ids, bags = d["ids"], _l2(d["bags"]).astype(np.float32)
    conn = db.connect()
    info = {r["id"]: r for r in conn.execute(
        "SELECT id,split,binary_label,patient_id FROM samples WHERE dataset=?", (dataset,))}
    split = np.array([info[int(i)]["split"] for i in ids])
    y = np.array([info[int(i)]["binary_label"] for i in ids])
    pat = np.array([info[int(i)]["patient_id"] for i in ids])
    return conn, ids, bags, y, split, pat


def train_model(X, y, D, epochs, lr, wd, H, dropout, pos_weight):
    m = GatedAttentionMIL(D, H, dropout).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    X = torch.tensor(X); y = torch.tensor(y, dtype=torch.float32)
    n, bs = len(X), 64
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = X[idx].to(DEVICE), y[idx].to(DEVICE)
            opt.zero_grad()
            loss = lossf(m(xb), yb)
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def predict(m, X):
    m.eval()
    return torch.sigmoid(m(torch.tensor(X).to(DEVICE))).cpu().numpy()


def _best_threshold(y, s):
    thr, best = 0.5, -1
    for t in np.unique(s):
        ba = balanced_accuracy_score(y, (s >= t).astype(int))
        if ba > best:
            best, thr = ba, float(t)
    return thr


def _metrics(y, s, thr):
    d = evaluate.compute_metrics(y, (s >= thr).astype(int), s)
    d["bal_acc"] = 0.5 * (d["sensitivity"] + d["specificity"])
    return d


def run(encoder, dataset="cbis", method="mean", H=128, dropout=0.4, lr=1e-3,
        wd=1e-3, epochs=50, seed=42, n_splits=5):
    conn, ids, bags, y, split, pat = load(encoder, dataset)
    D = bags.shape[2]
    tr, te = split == "train", split == "test"
    Xtr, ytr, ptr, idtr = bags[tr], y[tr], pat[tr], ids[tr]
    Xte, idte = bags[te], ids[te]
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                              dtype=torch.float32).to(DEVICE)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(Xtr))
    te_preds, cv_aucs = [], []
    for trf, vaf in sgkf.split(Xtr, ytr, groups=ptr):
        m = train_model(Xtr[trf], ytr[trf], D, epochs, lr, wd, H, dropout, pos_weight)
        oof[vaf] = predict(m, Xtr[vaf])
        te_preds.append(predict(m, Xte))
        if len(set(ytr[vaf].tolist())) > 1:
            cv_aucs.append(roc_auc_score(ytr[vaf], oof[vaf]))
    te_score = np.mean(te_preds, axis=0)

    # threshold from train out-of-fold CASE scores (honest)
    cf_tr = evaluate.case_fuse(conn, [dict(ids=idtr, score_pos=oof)], method=method)
    thr = _best_threshold(cf_tr["y_true"], cf_tr["y_score"])
    # test case scores
    cf = evaluate.case_fuse(conn, [dict(ids=idte, score_pos=te_score)], method=method)
    y_te, s_te, pidx = cf["y_true"], cf["y_score"], cf["paired_idx"]
    full = _metrics(y_te, s_te, thr)
    paired = _metrics(y_te[pidx], s_te[pidx], thr)

    print(f"  MIL[{encoder}] method={method} H={H} drop={dropout} ep={epochs}  "
          f"CV-image-AUC={np.mean(cv_aucs):.4f}  thr={thr:.3f}")
    print(f"    FULL(210):   acc={full['accuracy']:.4f} bal_acc={full['bal_acc']:.4f} "
          f"auc={full['auc']:.4f} sens={full['sensitivity']:.4f} spec={full['specificity']:.4f}")
    print(f"    PAIRED({len(pidx)}): acc={paired['accuracy']:.4f} bal_acc={paired['bal_acc']:.4f} "
          f"auc={paired['auc']:.4f} sens={paired['sensitivity']:.4f} spec={paired['specificity']:.4f}")
    return full, paired


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="dinov2_vitb14_noclahe")
    ap.add_argument("--dataset", default="cbis")
    ap.add_argument("--method", default="mean", choices=["mean", "max", "min", "noisy_or"])
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    args = ap.parse_args()
    run(args.embedder, args.dataset, method=args.method, H=args.H, dropout=args.dropout,
        lr=args.lr, wd=args.wd, epochs=args.epochs)
