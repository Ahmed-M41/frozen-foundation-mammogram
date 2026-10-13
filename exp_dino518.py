"""User test: give every extractor the full image at its BEST input size, CBIS BASE4.

On cbis_full all four already get the full image, no crop, no CLAHE. The only extractor
below its best is DINOv2 (224x224 -- a full mammogram crushed to a tiny square). This
re-encodes DINOv2 ViT-L/14 at 518x518 (its high-res mode, matching RAD-DINO) on the full
images and swaps it into BASE4, holding the other three as-is. Reports CBIS paired AUC
for BASE4 with DINOv2@224 (current) vs DINOv2@518, same probe and seeds.
"""
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

import db
import embedders as EMB
import miniddsm_B as M
from data_scaling_extend import head_case_scores, align_B, _l2
from exp_msfeat_fusion import cbis_ms

SEEDS = [0, 1, 2, 3]
POS = ["MALIGNANT", "CANCER"]


def meta():
    conn = db.connect()
    rows = conn.execute(
        "SELECT id,file_path,split,pathology,view,patient_id FROM samples "
        "WHERE dataset='cbis_full' AND view IN ('CC','MLO') ORDER BY id").fetchall()
    ids = np.array([r["id"] for r in rows])
    return (conn, ids, [r["file_path"] for r in rows],
            np.array([1 if str(r["pathology"]).upper() in POS else 0 for r in rows]),
            np.array([r["split"] for r in rows]),
            np.array([r["patient_id"] for r in rows]),
            np.array([r["view"] for r in rows]))


def cached(conn, name, ids):
    eid, v, _ = db.load_embeddings(conn, name, dataset=["cbis_full"])
    idx = {int(i): k for k, i in enumerate(eid)}
    return _l2(v[[idx[int(i)] for i in ids]]).astype(np.float32)


def encode_dino(size, paths, bs=16):
    e = EMB._dinov2("dino_hr", "dinov2_vitl14", size=size, use_clahe=False)
    out = []
    for i in range(0, len(paths), bs):
        out.append(e.embed_batch([Image.open(p) for p in paths[i:i+bs]]))
    return _l2(np.concatenate(out, 0)).astype(np.float32)


def probe_auc(conn, ids, X, y, grp, view, tr, te, ms):
    aucs = []
    for seed in SEEDS:
        rt, rte = head_case_scores(conn, ids, X, y, grp, view, tr, te, M.C_HEAD, seed)
        _, _, _, yte, Rte, _ = align_B(rt, rte, ms, ms)
        aucs.append(roc_auc_score(yte, Rte))
    return np.mean(aucs), np.std(aucs)


def main():
    conn, ids, paths, y, split, grp, view = meta()
    tr, te = split == "train", split == "test"
    ms = cbis_ms()
    mc = cached(conn, "mammo_clip_b5", ids)
    rd = cached(conn, "rad_dino_noclahe", ids)
    msf = cached(conn, "mammoscreen_feat", ids)
    d224 = cached(conn, "dinov2_vitl14_noclahe", ids)          # current (224)

    print("encoding DINOv2 @ 518 on full images ...", flush=True)
    d518 = encode_dino(518, paths)

    for tag, d in [("DINOv2 @ 224 (current)", d224), ("DINOv2 @ 518 (best)", d518)]:
        X4 = np.concatenate([mc, d, rd, msf], 1)               # order: MC, DINOv2, RAD, MammoScreen
        a, s = probe_auc(conn, ids, X4, y, grp, view, tr, te, ms)
        print(f"  BASE4 [{tag:24s}]  CBIS paired AUC = {a:.4f} +- {s:.4f}", flush=True)


if __name__ == "__main__":
    main()
