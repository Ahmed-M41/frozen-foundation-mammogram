"""Family 1, Option 2: attention-MIL + a TRAINED two-view (CC/MLO) fusion head.

Builds on Option 1: a shared gated-attention MIL turns each view's patch bag into
a representation; a small fusion MLP then combines the CC and MLO representations
into ONE case decision — learning the multi-view fusion instead of our hand-picked
mean/max. Backbone stays frozen. Same honest protocol: 5-fold patient-grouped CV
on train cases (out-of-fold sets the threshold; 5 models averaged on test).

Run:  python mil_twoview.py --embedder dinov2_vitl14_noclahe
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import db
import evaluate
from patch_embed import CACHE_DIR
from mil import GatedAttentionMIL, _l2, _best_threshold, _metrics, DEVICE


class TwoViewMIL(nn.Module):
    def __init__(self, D, H=192, dropout=0.6):
        super().__init__()
        self.mil = GatedAttentionMIL(D, H, dropout)
        self.fuse = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * D, D), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(D, 1))

    def forward(self, cc, mlo):
        r = torch.cat([self.mil.rep(cc), self.mil.rep(mlo)], dim=1)  # [B,2D]
        return self.fuse(r).squeeze(-1)


def load_cases(encoder, dataset="cbis"):
    d = np.load(os.path.join(CACHE_DIR, f"{encoder}__{dataset}.npz"))
    ids, bags = d["ids"], _l2(d["bags"]).astype(np.float32)
    conn = db.connect()
    info = {r["id"]: r for r in conn.execute(
        "SELECT id,split,binary_label,patient_id,view,case_id FROM samples WHERE dataset=?",
        (dataset,))}
    idx_of = {int(i): k for k, i in enumerate(ids)}
    cimg, cmeta = defaultdict(dict), {}
    for i in ids:
        r = info[int(i)]
        cimg[r["case_id"]][r["view"]] = idx_of[int(i)]
        cmeta[r["case_id"]] = (r["split"], r["binary_label"], r["patient_id"])
    cases = sorted(cimg)
    Xcc, Xmlo, y, pat, split, paired = [], [], [], [], [], []
    for c in cases:
        v = cimg[c]
        cc, mlo = v.get("CC"), v.get("MLO")
        paired.append(cc is not None and mlo is not None)
        if cc is None:
            cc = mlo
        if mlo is None:
            mlo = cc
        Xcc.append(bags[cc]); Xmlo.append(bags[mlo])
        sp, lab, p = cmeta[c]
        y.append(lab); pat.append(p); split.append(sp)
    return (conn, np.array(cases), np.stack(Xcc), np.stack(Xmlo),
            np.array(y), np.array(split), np.array(pat), np.array(paired))


def train_model(Xcc, Xmlo, y, D, epochs, lr, wd, H, dropout, pos_weight):
    m = TwoViewMIL(D, H, dropout).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xcc, Xmlo = torch.tensor(Xcc), torch.tensor(Xmlo)
    yt = torch.tensor(y, dtype=torch.float32)
    n, bs = len(Xcc), 64
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = m(Xcc[idx].to(DEVICE), Xmlo[idx].to(DEVICE))
            loss = lossf(out, yt[idx].to(DEVICE))
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def predict(m, Xcc, Xmlo):
    m.eval()
    return torch.sigmoid(m(torch.tensor(Xcc).to(DEVICE),
                           torch.tensor(Xmlo).to(DEVICE))).cpu().numpy()


def run(encoder, dataset="cbis", H=192, dropout=0.6, lr=1e-3, wd=1e-3, epochs=60,
        seed=42, n_splits=5):
    conn, cases, Xcc, Xmlo, y, split, pat, paired = load_cases(encoder, dataset)
    D = Xcc.shape[2]
    tr, te = split == "train", split == "test"
    pos_weight = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)],
                              dtype=torch.float32).to(DEVICE)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    Xcc_tr, Xmlo_tr, ytr, ptr = Xcc[tr], Xmlo[tr], y[tr], pat[tr]
    oof = np.zeros(tr.sum())
    te_preds, cv = [], []
    for trf, vaf in sgkf.split(Xcc_tr, ytr, groups=ptr):
        m = train_model(Xcc_tr[trf], Xmlo_tr[trf], ytr[trf], D, epochs, lr, wd, H, dropout, pos_weight)
        oof[vaf] = predict(m, Xcc_tr[vaf], Xmlo_tr[vaf])
        te_preds.append(predict(m, Xcc[te], Xmlo[te]))
        if len(set(ytr[vaf].tolist())) > 1:
            cv.append(roc_auc_score(ytr[vaf], oof[vaf]))
    te_score = np.mean(te_preds, axis=0)
    thr = _best_threshold(ytr, oof)

    yte, pte = y[te], paired[te]
    full = _metrics(yte, te_score, thr)
    pm = _metrics(yte[pte], te_score[pte], thr)
    print(f"  TwoViewMIL[{encoder}] H={H} drop={dropout} ep={epochs}  "
          f"CV-case-AUC={np.mean(cv):.4f}  thr={thr:.3f}")
    print(f"    FULL({te.sum()}):   acc={full['accuracy']:.4f} bal_acc={full['bal_acc']:.4f} "
          f"auc={full['auc']:.4f} sens={full['sensitivity']:.4f} spec={full['specificity']:.4f}")
    print(f"    PAIRED({pte.sum()}): acc={pm['accuracy']:.4f} bal_acc={pm['bal_acc']:.4f} "
          f"auc={pm['auc']:.4f} sens={pm['sensitivity']:.4f} spec={pm['specificity']:.4f}")
    return full, pm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="dinov2_vitl14_noclahe")
    ap.add_argument("--dataset", default="cbis")
    ap.add_argument("--H", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    args = ap.parse_args()
    run(args.embedder, args.dataset, H=args.H, dropout=args.dropout, lr=args.lr,
        wd=args.wd, epochs=args.epochs)
