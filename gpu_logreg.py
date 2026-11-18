"""GPU drop-in for multiview_probe.head_cv (EXPLORATORY runs only; paper numbers stay on sklearn).
Solves exactly sklearn's LogisticRegression(C, class_weight='balanced') problem after the same
StandardScaler: minimise 0.5*||w||^2 + C * sum_i s_i * logloss_i (intercept not penalised),
s_i = n / (2 * n_class(i)). Full-batch L-BFGS on the GPU (float32; MV_GPU_FP64=1 for float64).
Validated 2026-11-15: CBIS-DDSM MammoFF 0.902 (= paper), INbreast B 0.947 (paper 0.949).
Speed-up only 1.2-1.6x: the problem is ill-conditioned, so iteration count, not arithmetic,
dominates. Enable with MV_GPU=1 (multiview_probe.head_cv dispatches here)."""
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

DEV = "cuda" if torch.cuda.is_available() else "cpu"
import os
DT = torch.float32 if os.environ.get("MV_GPU_FP64") != "1" else torch.float64


def _fit_predict(Xf, yf, Xev_list, C, max_iter=500):
    X = torch.as_tensor(Xf, dtype=DT, device=DEV)
    y = torch.as_tensor(yf, dtype=DT, device=DEV)
    mu = X.mean(0); sd = X.std(0, unbiased=False); sd[sd == 0] = 1.0       # StandardScaler
    Xs = (X - mu) / sd
    n = len(y); n1 = y.sum(); n0 = n - n1
    s = torch.where(y == 1, n / (2 * n1), n / (2 * n0))                     # balanced weights
    w = torch.zeros(Xs.shape[1], dtype=DT, device=DEV, requires_grad=True)
    b = torch.zeros(1, dtype=DT, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=max_iter, tolerance_grad=1e-5,
                            tolerance_change=1e-9, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xs @ w + b
        loss = 0.5 * (w * w).sum() + C * (s * torch.nn.functional.softplus(-(2 * y - 1) * z)).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        out = []
        for Xe in Xev_list:
            Xe = (torch.as_tensor(Xe, dtype=DT, device=DEV) - mu) / sd
            out.append(torch.sigmoid(Xe @ w + b).cpu().numpy())
    return out


def head_cv(Xtr, ytr, ptr, Xte, C, seed=42, n_splits=5):
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof, tep = np.zeros(len(Xtr)), []
    for trf, vaf in sg.split(Xtr, ytr, groups=ptr):
        o, t = _fit_predict(Xtr[trf], ytr[trf], [Xtr[vaf], Xte], C)
        oof[vaf] = o; tep.append(t)
    return oof, np.mean(tep, axis=0)
