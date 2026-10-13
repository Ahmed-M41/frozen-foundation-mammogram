"""One-off experiment (user-requested): replace the frozen MammoScreen second
signal with keanteng/swin-v2-large-ft-breast-cancer-classification-0603 (fine-tuned
on a Mini-DDSM/DDSM-derived set, used AS-IS, no retraining) and re-run the exact
Mini-DDSM normal-vs-cancer fusion protocol from paper_numbers.py.

Reports, averaged over the paper's 4 seeds / 20% patient-disjoint splits:
  AUC(probe R alone) | AUC(second signal M alone) | AUC(fused w=0.6)
for BOTH the original MammoScreen and the keanteng model, so they are comparable.

NOTE the caveat: keanteng likely trained on overlapping Mini-DDSM images, so its
M-alone AUC here may be leak-inflated (that is the MammoScreen-on-CBIS analog).
"""
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

import config, db
import miniddsm_B as M
from data_scaling_extend import head_case_scores, align_B
from data_scaling import _pct
from mil import _best_threshold, _metrics

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REPO = "keanteng/swin-v2-large-ft-breast-cancer-classification-0603"
TF = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


CANCER_COL = 0        # config classes = ['Has_Cancer', 'Normal']


class KeanNet(torch.nn.Module):
    """swinv2-large backbone + the checkpoint's custom 2-class head."""
    def __init__(self):
        super().__init__()
        from transformers import Swinv2Model, Swinv2Config
        cfg = Swinv2Config.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
        self.swinv2 = Swinv2Model(cfg)                       # weights loaded from keanteng ckpt
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1536, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.0), torch.nn.Linear(256, 2))

    def forward(self, x):
        return self.classifier(self.swinv2(x).pooler_output)


def load_model():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    m = KeanNet()
    sd = load_file(hf_hub_download(REPO, "model.safetensors"))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:4]}"
    assert not [k for k in missing if not k.startswith(("swinv2.pooler",))], \
        f"missing backbone keys: {missing[:4]}"
    print(f"  loaded {REPO} | 2-class head, cancer col={CANCER_COL} "
          f"({len(sd)} tensors, {len(missing)} missing/{len(unexpected)} unexpected)")
    return m.eval().to(DEV)


@torch.no_grad()
def probs_for_paths(model, paths, bs=32):
    """Return softmax probs [N, C] for a list of image paths."""
    out = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(DEV)
        out.append(torch.softmax(model(batch), 1).cpu().numpy())
    return np.concatenate(out, 0)


def nc_case_probs(model):
    """Every Mini-DDSM normal/cancer case -> (case_id, label, meanprob[C]).
    label: CANCER=1, NORMAL=0.  Averages the softmax over the breast's CC+MLO views."""
    conn = db.connect()
    rows = conn.execute("""
        SELECT case_id,
               MAX(pathology) patho,
               MAX(CASE WHEN view='CC'  THEN file_path END) cc,
               MAX(CASE WHEN view='MLO' THEN file_path END) mlo
        FROM samples
        WHERE (dataset='miniddsm' AND pathology='CANCER')
           OR (dataset='miniddsm_normal' AND pathology='NORMAL')
        GROUP BY case_id
    """).fetchall()
    cids, ys, imgpaths, spans = [], [], [], []
    for r in rows:
        ps = [p for p in (r["cc"], r["mlo"]) if p]
        if not ps:
            continue
        cids.append(str(r["case_id"]))
        ys.append(1 if str(r["patho"]).upper() == "CANCER" else 0)
        spans.append((len(imgpaths), len(imgpaths) + len(ps)))
        imgpaths.extend(ps)
    P = probs_for_paths(model, imgpaths)
    caseP = np.array([P[a:b].mean(0) for (a, b) in spans])   # [Ncase, C]
    return np.array(cids), np.array(ys), caseP


def main():
    print("=== keanteng as-is second-signal experiment (Mini-DDSM normal/cancer) ===")
    model = load_model()
    cids, y, caseP = nc_case_probs(model)
    print(f"  scored {len(cids)} nc cases (pos/cancer={int(y.sum())}, neg/normal={int((y==0).sum())})")

    # data for the probe (same as paper_numbers nc block)
    conn = db.connect()
    ids, X, yy, split, grp, view, ds, patho, abn = M.apply_task(
        "nc", *M.load_base(conn, ["miniddsm", "miniddsm_normal"]))
    ms_mam = M.load_ms(["miniddsm", "miniddsm_normal"])              # MammoScreen map

    # pick the keanteng "cancer" class index from a fixed train-like reference:
    # use ALL cases' labels once to identify which of the C columns tracks cancer.
    # (choosing the column, not tuning a threshold -> not test-set oracle.)
    aucs = [roc_auc_score(y, caseP[:, c]) for c in range(caseP.shape[1])]
    cidx = int(np.argmax(aucs))
    print(f"  keanteng class AUCs (all-case, for column choice): "
          f"{[f'{a:.3f}' for a in aucs]} -> cancer col = {cidx}")
    kean_map = {c: float(caseP[i, cidx]) for i, c in enumerate(cids)}

    def run(ms_map, tag):
        RA, MA, FA, WS = [], [], [], []
        for seed in M.SEEDS:
            te = M.patient_test_mask(grp, yy, 0.2, seed)
            rcf_tr, rcf_te = head_case_scores(conn, ids, X, yy, grp, view, ~te, te, M.C_HEAD, seed)
            ytr, Rtr, Mtr, yte, Rte, Mte = align_B(rcf_tr, rcf_te, ms_map, ms_map)
            RA.append(roc_auc_score(yte, Rte))
            MA.append(roc_auc_score(yte, Mte))
            # fused at fixed w=0.6, honest train threshold
            Rp, Mp = _pct(Rtr, Rtr), _pct(Mtr, Mtr)
            Rpe, Mpe = _pct(Rte, Rtr), _pct(Mte, Mtr)
            fte = 0.6 * Rpe + 0.4 * Mpe
            FA.append(roc_auc_score(yte, fte))
            # best-w on train, for reference
            bw = max(np.linspace(0, 1, 21),
                     key=lambda w: roc_auc_score(ytr, w * Rp + (1 - w) * Mp))
            WS.append(roc_auc_score(yte, bw * Rpe + (1 - bw) * Mpe))
        f = lambda a: (np.mean(a), np.std(a))
        print(f"\n  [{tag}]  (mean over {len(M.SEEDS)} seeds)")
        print(f"    probe R alone   AUC = {f(RA)[0]:.3f} ± {f(RA)[1]:.3f}")
        print(f"    signal M alone  AUC = {f(MA)[0]:.3f} ± {f(MA)[1]:.3f}")
        print(f"    fused  w=0.6    AUC = {f(FA)[0]:.3f} ± {f(FA)[1]:.3f}")
        print(f"    fused  best-w   AUC = {f(WS)[0]:.3f} ± {f(WS)[1]:.3f}")
        return f(RA)[0], f(MA)[0], f(FA)[0]

    print("\n----------------------------------------------------------------")
    run(ms_mam, "MammoScreen (original, sanity check)")
    run(kean_map, "keanteng Swin-V2 (as-is)")
    print("----------------------------------------------------------------")
    print("Question: does either fused AUC beat the probe-alone AUC?")


if __name__ == "__main__":
    main()
