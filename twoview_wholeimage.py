"""Family 1, Option 2 on WHOLE-IMAGE features: a trained two-view (CC/MLO) MLP
head on frozen whole-image embeddings (e.g. Mammo-CLIP), instead of on weak
patch bags. Concatenates the CC and MLO feature vectors and learns the case
decision. Backbone frozen; honest 5-fold patient-CV + oof threshold + ensemble.

Run:  python twoview_wholeimage.py --encoders mammo_clip_b5 --dataset cbis_full
"""
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import db
from mil import _best_threshold, _metrics, DEVICE


def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


class TwoViewMLP(nn.Module):
    def __init__(self, D, H=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * D, H), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(H, 1))

    def forward(self, cc, mlo):
        return self.net(torch.cat([cc, mlo], dim=1)).squeeze(-1)


def load_cases(conn, encoders, dataset):
    ids0, mats = None, []
    for e in encoders:
        ids, v, _ = db.load_embeddings(conn, e, dataset=dataset)   # all splits
        if ids0 is None:
            ids0 = ids
        else:
            assert np.array_equal(ids, ids0), f"misalign {e}"
        mats.append(_l2(v))
    feats = np.concatenate(mats, 1).astype(np.float32)
    idx_of = {int(i): k for k, i in enumerate(ids0)}
    info = {r["id"]: r for r in conn.execute(
        "SELECT id,split,binary_label,patient_id,view,case_id FROM samples WHERE dataset=?",
        (dataset,))}
    cimg, cmeta = defaultdict(dict), {}
    for i in ids0:
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
        Xcc.append(feats[cc]); Xmlo.append(feats[mlo])
        sp, lab, p = cmeta[c]
        y.append(lab); pat.append(p); split.append(sp)
    return (np.stack(Xcc), np.stack(Xmlo), np.array(y), np.array(split),
            np.array(pat), np.array(paired))


def train_model(Xcc, Xmlo, y, D, epochs, lr, wd, H, dropout, pos_weight):
    m = TwoViewMLP(D, H, dropout).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xcc, Xmlo = torch.tensor(Xcc), torch.tensor(Xmlo)
    yt = torch.tensor(y, dtype=torch.float32)
    n, bs = len(Xcc), 32
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(m(Xcc[idx].to(DEVICE), Xmlo[idx].to(DEVICE)), yt[idx].to(DEVICE))
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def predict(m, Xcc, Xmlo):
    m.eval()
    return torch.sigmoid(m(torch.tensor(Xcc).to(DEVICE),
                           torch.tensor(Xmlo).to(DEVICE))).cpu().numpy()


def run(encoders, dataset="cbis_full", H=256, dropout=0.6, lr=5e-4, wd=1e-2,
        epochs=80, seed=42, n_splits=5):
    conn = db.connect()
    Xcc, Xmlo, y, split, pat, paired = load_cases(conn, encoders, dataset)
    D = Xcc.shape[1]
    tr, te = split == "train", split == "test"
    pw = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)],
                      dtype=torch.float32).to(DEVICE)
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    Xcc_tr, Xmlo_tr, ytr, ptr = Xcc[tr], Xmlo[tr], y[tr], pat[tr]
    oof = np.zeros(tr.sum())
    tep, cv = [], []
    for trf, vaf in sg.split(Xcc_tr, ytr, groups=ptr):
        m = train_model(Xcc_tr[trf], Xmlo_tr[trf], ytr[trf], D, epochs, lr, wd, H, dropout, pw)
        oof[vaf] = predict(m, Xcc_tr[vaf], Xmlo_tr[vaf])
        tep.append(predict(m, Xcc[te], Xmlo[te]))
        if len(set(ytr[vaf].tolist())) > 1:
            cv.append(roc_auc_score(ytr[vaf], oof[vaf]))
    te_score = np.mean(tep, axis=0)
    thr = _best_threshold(ytr, oof)
    yte, pte = y[te], paired[te]
    full = _metrics(yte, te_score, thr)
    pm = _metrics(yte[pte], te_score[pte], thr)
    print(f"  TwoViewMLP[{'+'.join(encoders)}] H={H} drop={dropout} ep={epochs}  "
          f"CV-case-AUC={np.mean(cv):.4f} thr={thr:.3f}")
    print(f"    FULL({te.sum()}):   acc={full['accuracy']:.4f} bal_acc={full['bal_acc']:.4f} "
          f"auc={full['auc']:.4f} sens={full['sensitivity']:.4f} spec={full['specificity']:.4f}")
    print(f"    PAIRED({pte.sum()}): acc={pm['accuracy']:.4f} bal_acc={pm['bal_acc']:.4f} "
          f"auc={pm['auc']:.4f} sens={pm['sensitivity']:.4f} spec={pm['specificity']:.4f}")
    return full, pm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--dataset", default="cbis_full")
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    args = ap.parse_args()
    run(args.encoders, dataset=args.dataset, H=args.H, dropout=args.dropout,
        epochs=args.epochs, lr=args.lr, wd=args.wd)
